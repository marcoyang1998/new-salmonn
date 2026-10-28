from types import SimpleNamespace

import pytest
import torch

from backbone_compat import (
    build_assistant_labels,
    clean_generated_output,
    get_decoder_layer_name,
    replace_audio_placeholders,
    resolve_tokenizer_capabilities,
    validate_saved_identity,
    validate_tokenizer_embedding_size,
)


class TinyTokenizer:
    def __init__(self, model_type="qwen2", thinking=False):
        self.model_type = model_type
        self.chat_template = "enable_thinking <think>" if thinking else "plain template"
        specials = (
            ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
            if model_type.startswith("qwen")
            else ["<|reserved_special_token_0|>", "<|reserved_special_token_1|>"]
        )
        self.vocab = {token: index + 1 for index, token in enumerate(specials)}
        self.vocab.update({chr(code): len(self.vocab) + code for code in range(32, 127)})
        self.unk_token_id = 9999
        self.eos_token_id = self.vocab.get("<|im_end|>", 2)
        self.pad_token_id = 0
        self.special_tokens_map = {"eos_token": "<|im_end|>"}

    def __len__(self):
        return max(self.vocab.values()) + 1

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def get_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=False):
        ids = []
        specials = sorted(
            (token for token in self.vocab if token.startswith("<|")), key=len, reverse=True
        )
        while text:
            match = next((token for token in specials if text.startswith(token)), None)
            if match:
                ids.append(self.vocab[match])
                text = text[len(match) :]
            else:
                ids.append(self.vocab.get(text[0], self.unk_token_id))
                text = text[1:]
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = torch.tensor([self.encode(text, add_special_tokens=add_special_tokens)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None
    ):
        if self.model_type.startswith("qwen"):
            text = "".join(
                f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n"
                for item in messages
            )
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
        else:
            text = "".join(f"<{item['role']}>\n{item['content']}<eot>" for item in messages)
            if add_generation_prompt:
                text += "<assistant>\n"
        return text


@pytest.mark.parametrize(
    ("model_type", "layer"),
    [
        ("qwen2", "Qwen2DecoderLayer"),
        ("qwen3", "Qwen3DecoderLayer"),
        ("qwen3_moe", "Qwen3MoeDecoderLayer"),
        ("llama", "LlamaDecoderLayer"),
    ],
)
def test_decoder_layer_mapping(model_type, layer):
    assert get_decoder_layer_name(SimpleNamespace(model_type=model_type)) == layer


def test_qwen2_capabilities_and_audio_expansion():
    tokenizer = TinyTokenizer("qwen2")
    capabilities = resolve_tokenizer_capabilities(
        tokenizer, SimpleNamespace(model_type="qwen2", eos_token_id=2, pad_token_id=0)
    )
    assert capabilities.model_type == "qwen2"
    assert not capabilities.supports_thinking
    assert replace_audio_placeholders("<audio>x<audio>", [2, 1], capabilities) == (
        "<|vision_start|><|vision_start|><|vision_end|>x"
        "<|vision_start|><|vision_end|>"
    )


def test_assistant_labels_cover_only_answer_and_terminator():
    tokenizer = TinyTokenizer("qwen2")
    capabilities = resolve_tokenizer_capabilities(
        tokenizer, SimpleNamespace(model_type="qwen2", eos_token_id=2, pad_token_id=0)
    )
    messages = [
        {"role": "user", "content": "<audio>question"},
        {"role": "assistant", "content": "answer"},
    ]
    encoded, labels, _ = build_assistant_labels(tokenizer, messages, [1], capabilities)
    labeled_ids = labels[labels != -100].tolist()
    assert tokenizer.encode("answer<|im_end|>\n") == labeled_ids
    assert capabilities.audio_start_token_id not in labeled_ids
    assert encoded["input_ids"].shape == labels.unsqueeze(0).shape


def test_output_cleanup_is_capability_driven():
    plain = resolve_tokenizer_capabilities(
        TinyTokenizer("qwen2"), SimpleNamespace(model_type="qwen2", eos_token_id=2, pad_token_id=0)
    )
    thinking = resolve_tokenizer_capabilities(
        TinyTokenizer("qwen3", thinking=True),
        SimpleNamespace(model_type="qwen3", eos_token_id=2, pad_token_id=0),
    )
    assert clean_generated_output("literal </think> answer", plain) == "literal </think> answer"
    assert clean_generated_output("reasoning </think> answer", thinking) == "answer"


def test_new_identity_rejects_mismatched_tokenizer():
    capabilities = resolve_tokenizer_capabilities(
        TinyTokenizer("qwen2"), SimpleNamespace(model_type="qwen2", eos_token_id=2, pad_token_id=0)
    )
    saved = capabilities.metadata()
    saved["tokenizer_length"] += 1
    with pytest.raises(ValueError, match="tokenizer_length"):
        validate_saved_identity(saved, capabilities)


def test_transformers_qwen2_dispatch():
    from transformers import AutoModelForCausalLM, Qwen2Config

    config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    model = AutoModelForCausalLM.from_config(config)
    assert type(model).__name__ == "Qwen2ForCausalLM"
    assert type(model.model.layers[0]).__name__ == "Qwen2DecoderLayer"


def test_reserved_embedding_rows_are_allowed():
    tokenizer = TinyTokenizer("qwen2")
    model = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(num_embeddings=len(tokenizer) + 399)
    )
    validate_tokenizer_embedding_size(tokenizer, model)
