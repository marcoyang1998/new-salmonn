"""Backbone/tokenizer compatibility helpers shared by training and inference.

Keep this module independent of SALMONN's audio dependencies so its behavior can
be covered by fast unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence


SUPPORTED_MODEL_TYPES = {"qwen2", "qwen3", "qwen3_moe", "llama"}
DECODER_LAYER_BY_MODEL_TYPE = {
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "qwen3_moe": "Qwen3MoeDecoderLayer",
    "llama": "LlamaDecoderLayer",
}


@dataclass(frozen=True)
class TokenizerCapabilities:
    model_type: str
    audio_start_token: str
    audio_end_token: str
    audio_start_token_id: int
    audio_end_token_id: int
    im_start_token_id: Optional[int]
    im_end_token_id: Optional[int]
    eos_token_ids: tuple[int, ...]
    pad_token_id: int
    supports_thinking: bool
    tokenizer_class: str
    tokenizer_length: int
    fingerprint: str

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        result["eos_token_ids"] = list(self.eos_token_ids)
        return result


def get_model_type(config: Any) -> str:
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported language-backbone model_type={model_type!r}; "
            f"expected one of {sorted(SUPPORTED_MODEL_TYPES)}."
        )
    return model_type


def get_decoder_layer_name(config: Any) -> str:
    return DECODER_LAYER_BY_MODEL_TYPE[get_model_type(config)]


def template_supports_thinking(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", "") or ""
    return any(marker in template for marker in ("enable_thinking", "<think>", "</think>"))


def _single_token_id(tokenizer: Any, token: str, *, required: bool = True) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    encoded = tokenizer.encode(token, add_special_tokens=False)
    valid = token_id is not None and token_id != unk_id and encoded == [token_id]
    if not valid:
        if required:
            raise ValueError(
                f"Required special token {token!r} must map to exactly one non-UNK token; "
                f"convert_tokens_to_ids={token_id!r}, encoded={encoded!r}, unk={unk_id!r}."
            )
        return None
    return int(token_id)


def normalize_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(dict.fromkeys(int(item) for item in value if item is not None))
    raise TypeError(f"Token IDs must be an integer or iterable of integers, got {type(value).__name__}.")


def tokenizer_fingerprint(tokenizer: Any, special_ids: Mapping[str, Any]) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "length": len(tokenizer),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "special_ids": dict(special_ids),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_tokenizer_capabilities(tokenizer: Any, config: Any) -> TokenizerCapabilities:
    model_type = get_model_type(config)
    if model_type.startswith("qwen"):
        audio_start_token, audio_end_token = "<|vision_start|>", "<|vision_end|>"
        im_start = _single_token_id(tokenizer, "<|im_start|>", required=True)
        im_end = _single_token_id(tokenizer, "<|im_end|>", required=True)
    else:
        audio_start_token = "<|reserved_special_token_0|>"
        audio_end_token = "<|reserved_special_token_1|>"
        im_start = im_end = None

    audio_start = _single_token_id(tokenizer, audio_start_token, required=True)
    audio_end = _single_token_id(tokenizer, audio_end_token, required=True)
    if audio_start == audio_end:
        raise ValueError("Audio start and end sentinels must have distinct token IDs.")

    eos_ids = normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    if not eos_ids:
        eos_ids = normalize_token_ids(getattr(config, "eos_token_id", None))
    if not eos_ids:
        raise ValueError("Tokenizer/model does not define a usable eos_token_id.")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(config, "pad_token_id", None)
    if pad_id is None:
        raise ValueError("Tokenizer/model does not define a usable pad_token_id.")

    ids = {
        "audio_start": audio_start,
        "audio_end": audio_end,
        "im_start": im_start,
        "im_end": im_end,
        "eos": eos_ids,
        "pad": pad_id,
    }
    return TokenizerCapabilities(
        model_type=model_type,
        audio_start_token=audio_start_token,
        audio_end_token=audio_end_token,
        audio_start_token_id=audio_start,
        audio_end_token_id=audio_end,
        im_start_token_id=im_start,
        im_end_token_id=im_end,
        eos_token_ids=eos_ids,
        pad_token_id=int(pad_id),
        supports_thinking=template_supports_thinking(tokenizer),
        tokenizer_class=tokenizer.__class__.__name__,
        tokenizer_length=len(tokenizer),
        fingerprint=tokenizer_fingerprint(tokenizer, ids),
    )


def validate_tokenizer_embedding_size(tokenizer: Any, model: Any) -> None:
    embeddings = model.get_input_embeddings()
    embedding_count = int(embeddings.num_embeddings)
    tokenizer_size = len(tokenizer)
    vocab = tokenizer.get_vocab()
    max_token_id = max(vocab.values()) if vocab else -1
    if tokenizer_size > embedding_count or max_token_id >= embedding_count:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer_size}, max ID {max_token_id}) does not fit the "
            f"language-model embedding table ({embedding_count}). Use the tokenizer saved with "
            "this checkpoint."
        )
    if tokenizer_size != embedding_count:
        logging.warning(
            "Tokenizer has %s entries while the embedding table has %s rows; the %s unused "
            "rows are reserved by the backbone.",
            tokenizer_size,
            embedding_count,
            embedding_count - tokenizer_size,
        )


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def replace_audio_placeholders(
    text: str,
    audio_nums: Sequence[int],
    capabilities: TokenizerCapabilities,
) -> str:
    expected = text.count("<audio>")
    if expected != len(audio_nums):
        raise ValueError(
            f"Audio placeholder count ({expected}) does not match audio group count ({len(audio_nums)})."
        )
    for audio_num in audio_nums:
        if capabilities.model_type == "llama":
            # Preserve the established Llama prompt representation.
            replacement = capabilities.audio_start_token + capabilities.audio_end_token
        else:
            replacement = capabilities.audio_start_token * int(audio_num) + capabilities.audio_end_token
        text = text.replace("<audio>", replacement, 1)
    return text


def build_assistant_labels(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    audio_nums: Sequence[int],
    capabilities: TokenizerCapabilities,
    *,
    skip_thinking: bool = False,
) -> tuple[Any, Any, str]:
    """Tokenize a training chat and label only the assistant response suffix."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Training chat must end with an assistant message.")
    enable_thinking = False if skip_thinking and capabilities.supports_thinking else None
    full_text = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    prompt_text = render_chat(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = replace_audio_placeholders(full_text, audio_nums, capabilities)
    prompt_text = replace_audio_placeholders(prompt_text, audio_nums, capabilities)
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    prompt = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = full["input_ids"][0]
    prompt_ids = prompt["input_ids"][0]
    if len(prompt_ids) >= len(input_ids) or not input_ids[: len(prompt_ids)].equal(prompt_ids):
        raise ValueError(
            "Chat template is not incrementally prefix-stable for user+assistant training; "
            "refusing to construct a potentially incorrect loss mask."
        )
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100
    if not (labels != -100).any():
        raise ValueError("Assistant response produced no labeled tokens.")
    return full, labels, full_text


def clean_generated_output(text: str, capabilities: TokenizerCapabilities) -> str:
    text = text.strip()
    if capabilities.supports_thinking and "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def validate_saved_identity(saved: Optional[Mapping[str, Any]], current: TokenizerCapabilities) -> None:
    if not saved:
        logging.warning("Checkpoint has no backbone_identity metadata; using legacy compatibility validation.")
        return
    for key, actual in (
        ("model_type", current.model_type),
        ("tokenizer_class", current.tokenizer_class),
        ("tokenizer_length", current.tokenizer_length),
        ("fingerprint", current.fingerprint),
    ):
        expected = saved.get(key)
        if expected is not None and expected != actual:
            raise ValueError(f"Checkpoint tokenizer identity mismatch for {key}: saved={expected!r}, current={actual!r}.")


def log_backbone_summary(config: Any, capabilities: TokenizerCapabilities) -> None:
    architectures = getattr(config, "architectures", None)
    logging.warning(
        "Resolved backbone: model_type=%s architectures=%s hidden_size=%s tokenizer=%s "
        "vocab=%s audio_ids=(%s,%s) im_ids=(%s,%s) thinking=%s eos=%s pad=%s",
        capabilities.model_type,
        architectures,
        getattr(config, "hidden_size", None),
        capabilities.tokenizer_class,
        capabilities.tokenizer_length,
        capabilities.audio_start_token_id,
        capabilities.audio_end_token_id,
        capabilities.im_start_token_id,
        capabilities.im_end_token_id,
        capabilities.supports_thinking,
        capabilities.eos_token_ids,
        capabilities.pad_token_id,
    )
