import os
import sys

from transformers import PreTrainedModel, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria, AutoFeatureExtractor
from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.reasoning_network import ReasoningNetwork

BERT_CKPT="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/google-bert--bert-base-uncased"

def is_peft_model(model):
    return getattr(model, "peft_config", None) is not None

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.stopped = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        B, sample_len = input_ids.shape
        if self.stopped is None:
            self.stopped = torch.zeros(B, dtype=torch.bool).to(input_ids.device)

        if self.stopped.all():
            return True

        unfinished_idxs = ~self.stopped
        check_sample = input_ids[unfinished_idxs]
        
        for stop in self.stops:
            n = len(stop)
            if sample_len < n:
                continue
            match = (stop == check_sample[:, -n:]).all(dim = 1)
            self.stopped[unfinished_idxs] |= match

        return False
    
class SALMONN(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def __init__(self, config, model_args):
        super().__init__(config)

        self.config = config
        self.encoder_type = model_args.encoder_type
        self.freeze_encoder = model_args.freeze_encoder
        self.llm_type = model_args.llm_type
        if self.llm_type == "Llama":
            self._no_split_modules =  ["LlamaDecoderLayer"]
        elif self.llm_type == "Qwen":
            self._no_split_modules = ["Qwen3DecoderLayer"]

        if model_args.base_llm_path:
            self.base_llm = AutoModelForCausalLM.from_pretrained(
                model_args.base_llm_path,
                dtype=config.torch_dtype,
                use_safetensors=True
            )
        else:
            self.base_llm = AutoModelForCausalLM.from_config(config)

        expand_vocab = getattr(model_args, "expand_vocab", False)
        freeze_llm = getattr(model_args, "freeze_llm", False)

        dora = getattr(model_args, "dora", False)
        assert not (freeze_llm and model_args.lora), \
            "freeze_llm=True is incompatible with lora=True."
        assert not (freeze_llm and dora), \
            "freeze_llm=True is incompatible with dora=True."
        assert not (model_args.lora and dora), \
            "lora=True and dora=True are mutually exclusive."
        assert not (freeze_llm and expand_vocab), \
            "freeze_llm=True is incompatible with expand_vocab=True."

        if expand_vocab:
            # base_llm is always loaded from the original (unexpanded) base LLM checkpoint,
            # so we can unconditionally resize by the fixed number of new tokens.
            original_vocab_size = self.base_llm.config.vocab_size
            target_vocab_size = original_vocab_size + len(self.NEW_SPECIAL_TOKENS)
            self.base_llm.resize_token_embeddings(target_vocab_size, mean_resizing=False)
            print(
                f"[SALMONN.__init__] expand_vocab=True: resized embeddings "
                f"{original_vocab_size} -> {target_vocab_size}"
            )

        if model_args.lora or dora:
            for name, param in self.base_llm.named_parameters():
                param.requires_grad = False
            lora_kwargs = dict(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                use_dora=dora,
            )
            if expand_vocab:
                lora_kwargs["modules_to_save"] = ["embed_tokens", "lm_head"]
                self.base_llm.config.tie_word_embeddings = False
            self.base_llm = get_peft_model(self.base_llm, LoraConfig(**lora_kwargs))
        elif expand_vocab:
            # No LoRA: freeze the entire LLM body except embed_tokens and lm_head,
            # which must be trainable so the new token rows receive gradient updates.
            for name, param in self.base_llm.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        elif freeze_llm:
            for param in self.base_llm.parameters():
                param.requires_grad_(False)
            print("[SALMONN.__init__] freeze_llm=True: all base_llm parameters frozen.")
        
        self.concat_encoder_features = getattr(model_args, "concat_encoder_features", False)
        
        if self.encoder_type == "dasheng_wavlm":
            self.audio_encoder, self.speech_encoder = self.get_audio_encoder(model_args)
            self.fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng", trust_remote_code=True)
        elif self.encoder_type == "whisper_beats":
            self.audio_encoder, self.speech_encoder = self.get_audio_encoder(model_args)
            self._dynamic_tied_weights_keys = [
                "audio_encoder.encoder.layers.*.self_attn.relative_attention_bias.weight"
            ] # modified
        else:
            self.audio_encoder = self.get_audio_encoder(model_args)
        if model_args.audio_encoder_path:
            if self.encoder_type == "zipformer2":
                import math
                from spear_encoder.scaling import ScaledLinear_lora
                info = self.audio_encoder.load_state_dict(
                    torch.load(model_args.audio_encoder_path, map_location="cpu")["model"],
                    strict=False,
                )
                print(f"Loading info: {info}")
                # lora_A / lora_B are not in the SpEAR checkpoint.
                # With HuggingFace low_cpu_mem_usage=True, missing parameters are
                # materialised as torch.empty (uninitialized / NaN).  Re-initialise
                # them here so that the subsequent eval() call (which merges lora
                # into weight) does not corrupt the loaded weights.
                # Also force merged=False so the merge/unmerge accounting is correct.
                for module in self.audio_encoder.modules():
                    if isinstance(module, ScaledLinear_lora) and module.r > 0:
                        with torch.no_grad():
                            nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                            nn.init.zeros_(module.lora_B)
                        module.merged = False
            elif self.encoder_type == "spear_transformer":
                info = self.audio_encoder.load_state_dict(
                    torch.load(model_args.audio_encoder_path, map_location="cpu")["model"],
                    strict=False,
                )
                print(f"Loading info: {info}")
        if model_args.freeze_encoder:
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            if self.encoder_type == "dasheng_wavlm" or self.encoder_type == "whisper_beats":
                for name, param in self.speech_encoder.named_parameters():
                    param.requires_grad = False
                self.speech_encoder.eval()
        encoder_lora = getattr(model_args, "encoder_lora", False)
        if model_args.encoder_type == "zipformer2" and encoder_lora:
            for name, param in self.audio_encoder.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            # we decided not to set to eval()
            # self.audio_encoder.eval()
        # _encoder_frozen can be toggled at runtime (e.g. by EncoderUnfreezeCallback)
        # without changing requires_grad, using torch.set_grad_enabled in encode_audio.
        self._encoder_frozen = model_args.freeze_encoder
        if self.encoder_type == "zipformer2":
            encoder_dim = self.audio_encoder.encoder_dim
            num_encoder_layers = sum(self.audio_encoder.encoder.num_encoder_layers)
            if self.concat_encoder_features:
                # assert not self.weighted_sum_encoder, "Cannot concat encoder features when using weighted sum"
                num_encoder_layers = self.audio_encoder.num_encoder_layers
                self.concat_proj = nn.Linear(int(num_encoder_layers * encoder_dim), encoder_dim)
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "spear_transformer":
            encoder_dim = self.audio_encoder.encoder_dim
            if self.concat_encoder_features:
                num_encoder_layers = self.audio_encoder.num_encoder_layers + 1
                self.concat_proj = nn.Linear(int(num_encoder_layers * encoder_dim), encoder_dim)
            self.ln_audio = nn.LayerNorm(encoder_dim)
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "dasheng":
            encoder_dim = self.audio_encoder.config.encoder_kwargs["embed_dim"]
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "dasheng_wavlm":
            encoder_dim = self.audio_encoder.config.encoder_kwargs["embed_dim"] + 2 * self.speech_encoder.config.output_hidden_size
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "whisper_beats":
            encoder_dim = self.audio_encoder.cfg.encoder_embed_dim + self.speech_encoder.config.d_model
            self.ln_audio = nn.LayerNorm(self.audio_encoder.cfg.encoder_embed_dim)
            self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        elif self.encoder_type == "whisper":
            encoder_dim = self.audio_encoder.config.d_model
            self.ln_audio = nn.LayerNorm(self.audio_encoder.config.d_model)
        elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
            encoder_dim = self.audio_encoder.config.output_dim
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "mimo":
            encoder_dim = self.audio_encoder.config.d_model
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "perception_av":
            encoder_dim = self.audio_encoder.audio_transformer.config.hidden_size # 1792
            self.ln_audio = nn.LayerNorm(encoder_dim)
        elif self.encoder_type == "audio_flamingo":
            encoder_dim = self.audio_encoder.config.d_model
            self.ln_audio = nn.LayerNorm(encoder_dim)

        self.connector_type = model_args.connector_type
        if model_args.connector_type == "MLP":
            self.connector_seg_size = model_args.connector_seg_size
            self.connector_hid_size = model_args.connector_hid_size
            self.connector = nn.Sequential(
                nn.Linear(encoder_dim * self.connector_seg_size, self.connector_hid_size),
                nn.ReLU(),
                nn.Linear(self.connector_hid_size, config.hidden_size)
            )
        elif model_args.connector_type == "Qformer":
            # TODO: make the following adjustable
            self.num_speech_query_token = getattr(model_args, "num_speech_query_token", 1) 
            self.second_per_window = getattr(model_args, "second_per_window", 0.333333)
            self.second_stride = getattr(model_args, "second_stride", 0.333333)
            
            encoder_dim = self.audio_encoder.encoder_dim # TODO: adjust for other audio encoder, currently only support SPEAR
            self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                num_query_token=self.num_speech_query_token, speech_width=encoder_dim
            )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            # Patch get_output_embeddings so that tie_weights() during from_pretrained()
            # does not crash when cls is None.
            self.speech_Qformer.get_output_embeddings = lambda: None
            
            freeze_speech_QFormer = getattr(model_args, "freeze_speech_QFormer", False)
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                print("freeze Speech QFormer")
            
            self.post_qformer_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, config.hidden_size
            )
            freeze_post_qformer_proj = getattr(model_args, "freeze_post_qformer_proj", False)
            if freeze_post_qformer_proj:
                for name, param in self.post_qformer_proj.named_parameters():
                    param.requires_grad = False
                self.post_qformer_proj.eval()
                print("freeze post QFormer proj")
        else:
            raise ValueError(f"Unsupported connector type: {model_args.connector_type}")

        # ---------- pause thinking ----------
        self.num_pause_steps = getattr(model_args, "num_pause_steps", 0)
        self.distinct_pause_embed = getattr(model_args, "distinct_pause_embed", False)
        self.use_reasoning_network = getattr(model_args, "use_reasoning_network", False)
        if self.num_pause_steps > 0:
            if self.use_reasoning_network:
                num_layers = getattr(model_args, "reasoning_network_num_layers", 5)
                reasoning_network_dim = getattr(model_args, "reasoning_network_dim", 1024)
                self.reasoning_network = ReasoningNetwork(
                    num_queries=self.num_pause_steps,
                    d_model=reasoning_network_dim,
                    num_layers=num_layers,
                    text_embed_dim=config.hidden_size,
                    audio_embed_dim=config.hidden_size,
                )
                self.post_reasoning_proj = nn.Linear(self.reasoning_network.d_model, config.hidden_size)
                print(f"A total of {sum(p.numel() for p in self.reasoning_network.parameters() if p.requires_grad)} trainable parameters in the reasoning network.")
                print(f"[SALMONN] reasoning_network enabled, num_queries={self.num_pause_steps}")
            else:
                if self.distinct_pause_embed:
                    self.pause_embed = nn.Parameter(torch.zeros(self.num_pause_steps, config.hidden_size))
                else:
                    self.pause_embed = nn.Parameter(torch.zeros(1, config.hidden_size))
                print(f"[SALMONN] pause_thinking enabled, num_pause_steps={self.num_pause_steps}, distinct_pause_embed={self.distinct_pause_embed}")

        # ---------- temporal embedding injection ----------
        self.inject_temporal_embedding = getattr(model_args, "inject_temporal_embedding", False)
        self.temporal_granularity = getattr(model_args, "temporal_granularity", 0.5)
        if self.inject_temporal_embedding:
            assert self.connector_type == "MLP", "inject_temporal_embedding is only supported with connector_type=MLP"
            encoder_frame_rate = getattr(model_args, "encoder_frame_rate", 50)
            self.output_frame_rate = encoder_frame_rate / self.connector_seg_size  # Hz after MLP connector
            frames_per_stamp = self.temporal_granularity * self.output_frame_rate
            assert abs(frames_per_stamp - round(frames_per_stamp)) < 1e-6, (
                f"temporal_granularity ({self.temporal_granularity}s) must be a multiple of the output frame "
                f"period (1/{self.output_frame_rate:.4g}s). Got {frames_per_stamp:.6f} frames per stamp, "
                f"which is not an integer."
            )
            # timestamp_token_ids is populated later via register_temporal_tokens(tokenizer)
            self.timestamp_token_ids = None
            print(
                f"[SALMONN] inject_temporal_embedding=True, granularity={self.temporal_granularity}s, "
                f"output_frame_rate={self.output_frame_rate:.1f}Hz, frames_per_stamp={round(frames_per_stamp)}"
            )

        # ---------- NL temporal embedding injection ----------
        self.inject_temporal_embedding_nl = getattr(model_args, "inject_temporal_embedding_nl", False)
        if self.inject_temporal_embedding_nl:
            assert not self.inject_temporal_embedding, (
                "inject_temporal_embedding_nl and inject_temporal_embedding cannot both be True."
            )
            assert not getattr(model_args, "expand_vocab", False), (
                "inject_temporal_embedding_nl and expand_vocab cannot both be True."
            )
            assert self.connector_type == "MLP", "inject_temporal_embedding_nl requires connector_type=MLP"
            encoder_frame_rate = getattr(model_args, "encoder_frame_rate", 50)
            self.output_frame_rate = encoder_frame_rate / self.connector_seg_size
            frames_per_stamp = self.temporal_granularity * self.output_frame_rate
            assert abs(frames_per_stamp - round(frames_per_stamp)) < 1e-6, (
                f"temporal_granularity ({self.temporal_granularity}s) must be a multiple of the output "
                f"frame period (1/{self.output_frame_rate:.4g}s). Got {frames_per_stamp:.6f} frames per stamp, "
                f"which is not an integer."
            )
            self.nl_timestamp_token_ids_list = None   # populated by register_nl_timestamp_tokenizer
            print(
                f"[SALMONN] inject_temporal_embedding_nl=True, granularity={self.temporal_granularity}s, "
                f"output_frame_rate={self.output_frame_rate:.1f}Hz, frames_per_stamp={round(self.temporal_granularity * self.output_frame_rate)}"
            )

    # ---------- vocabulary expansion ----------

    # 601 tokens: <|0.00|>, <|0.10|>, ..., <|60.00|>  (0.1 s granularity, 0–60 s)
    TIMESTAMP_TOKENS = [f"<|{i*0.1:.2f}|>" for i in range(601)]
    SPEAKER_TOKENS = ["<|speaker1|>", "<|speaker2|>", "<|speaker3|>", "<|speaker4|>", "<|speaker5|>"]
    NEW_SPECIAL_TOKENS = TIMESTAMP_TOKENS + SPEAKER_TOKENS

    def register_temporal_tokens(self, tokenizer):
        """Store a map from TIMESTAMP_TOKENS index -> token_id for use in inject_temporal_embeddings.

        Must be called after the tokenizer has been expanded with expand_llm_vocab (or loaded
        from a checkpoint that already contains the timestamp tokens).
        """
        ids = tokenizer.convert_tokens_to_ids(self.TIMESTAMP_TOKENS)
        self.timestamp_token_ids = ids  # plain list of length 601
        print(
            f"[SALMONN] register_temporal_tokens: mapped {len(ids)} timestamp tokens, "
            f"e.g. <|0.50|> -> {ids[5]}, <|1.00|> -> {ids[10]}"
        )

    def inject_temporal_embeddings(self, audio_embeds, audio_embeds_lens):
        """Interleave timestamp token embeddings into audio_embeds at temporal_granularity intervals.

        Every K audio frames one timestamp embedding is appended, giving blocks of (K+1):
            [a1 … a_K  <|t1|>]  [a_{K+1} … a_{2K}  <|t2|>]  …

        Steps:
          1. Pad seqlen to a multiple of K.
          2. Reshape to (bsz, num_blocks, K, dim).
          3. Build timestamp embeddings (num_blocks, 1, dim) and concatenate → (bsz, num_blocks, K+1, dim).
          4. Reshape to (bsz, num_blocks*(K+1), dim).

        Args:
            audio_embeds:      (bsz, seqlen, dim)
            audio_embeds_lens: (bsz,) valid frame counts (int64)

        Returns:
            new_audio_embeds:      (bsz, num_blocks*(K+1), dim)
            new_audio_embeds_lens: (bsz,)
        """
        assert self.timestamp_token_ids is not None, (
            "Call register_temporal_tokens(tokenizer) before using inject_temporal_embedding."
        )
        K = max(1, round(self.temporal_granularity * self.output_frame_rate))  # audio frames per stamp

        bsz, seqlen, dim = audio_embeds.shape
        num_blocks = (seqlen + K - 1) // K  # ceil(seqlen / K)

        # 1. Pad to multiple of K
        pad_len = num_blocks * K - seqlen
        if pad_len > 0:
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, pad_len))  # (bsz, num_blocks*K, dim)

        # 2. Reshape into blocks
        audio_embeds = audio_embeds.view(bsz, num_blocks, K, dim)  # (bsz, num_blocks, K, dim)

        # 3. Build one timestamp embedding per block: block i gets <|(i+1)*granularity|>
        embed_tokens = self.base_llm.get_input_embeddings()
        block_indices = torch.arange(num_blocks, device=audio_embeds.device)
        ts_times = (block_indices + 1).float() * self.temporal_granularity
        ts_list_indices = ts_times.div(0.1).round().long().clamp(0, len(self.timestamp_token_ids) - 1)
        ts_token_ids = torch.tensor(
            self.timestamp_token_ids, dtype=torch.long, device=audio_embeds.device
        )[ts_list_indices]                                                    # (num_blocks,)
        ts_embeds = embed_tokens(ts_token_ids).to(audio_embeds.dtype)        # (num_blocks, dim)
        ts_embeds = ts_embeds.unsqueeze(0).unsqueeze(2).expand(bsz, -1, 1, -1)  # (bsz, num_blocks, 1, dim)

        # 4. Concatenate and flatten: [a1…aK | ts] per block
        interleaved = torch.cat([audio_embeds, ts_embeds], dim=2)            # (bsz, num_blocks, K+1, dim)
        interleaved = interleaved.reshape(bsz, num_blocks * (K + 1), dim)

        new_audio_embeds_lens = audio_embeds_lens + (audio_embeds_lens + K - 1) // K
        return interleaved, new_audio_embeds_lens

    def register_nl_timestamp_tokenizer(self, tokenizer):
        """Pre-tokenise '<X.X seconds>' strings (0.0–120.0 s at 0.1 s steps) for NL timestamp injection.

        Stores token IDs for every possible timestamp as a list of lists so that forward-pass
        tokenisation is avoided entirely.  Must be called before using inject_temporal_embedding_nl.
        """
        token_ids_list = []
        for i in range(1201):                       # index 0 → 0.0 s, index 10 → 1.0 s, …, index 1200 → 120.0 s
            ts_time = round(i * 0.1, 1)
            ts_str = f"<{ts_time:.1f} seconds>"
            ids = tokenizer(ts_str, add_special_tokens=False)["input_ids"]
            token_ids_list.append(ids)
        self.nl_timestamp_token_ids_list = token_ids_list
        sample = token_ids_list[10]                 # "<1.0 seconds>"
        print(
            f"[SALMONN] register_nl_timestamp_tokenizer: "
            f"'<1.0 seconds>' -> {len(sample)} tokens {sample}"
        )

    def inject_temporal_embeddings_with_natural_language(self, audio_embeds, audio_embeds_lens):
        """Interleave natural-language timestamp embeddings into audio_embeds.

        Every K audio frames, a sequence of embeddings for '<T seconds>' is appended,
        where T = (block_index + 1) * temporal_granularity.  No vocabulary expansion is
        required — the base LLM tokenizer encodes the string as regular sub-word tokens.

        Block layout:
            [a_1 … a_K | <T_1 seconds> tokens] [a_{K+1} … a_{2K} | <T_2 seconds> tokens] …

        Args:
            audio_embeds:      (bsz, seqlen, dim)
            audio_embeds_lens: (bsz,) valid frame counts (int64)

        Returns:
            new_audio_embeds:      (bsz, total_len, dim)
            new_audio_embeds_lens: (bsz,)
        """
        assert self.nl_timestamp_token_ids_list is not None, (
            "Call register_nl_timestamp_tokenizer(tokenizer) before using inject_temporal_embedding_nl."
        )
        K = max(1, round(self.temporal_granularity * self.output_frame_rate))  # audio frames per stamp

        bsz, seqlen, dim = audio_embeds.shape
        num_blocks = (seqlen + K - 1) // K  # ceil(seqlen / K)

        # 1. Pad seqlen to multiple of K
        pad_len = num_blocks * K - seqlen
        if pad_len > 0:
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, pad_len))

        # 2. Pre-compute timestamp embeddings for every block
        embed_tokens = self.base_llm.get_input_embeddings()
        block_ts_embs = []   # list of (ts_len_i, dim) tensors
        ts_lens = []
        for i in range(num_blocks):
            ts_time = round((i + 1) * self.temporal_granularity, 10)
            ts_idx = int(round(ts_time / 0.1))
            ts_idx = max(0, min(ts_idx, len(self.nl_timestamp_token_ids_list) - 1))
            ids = torch.tensor(
                self.nl_timestamp_token_ids_list[ts_idx],
                dtype=torch.long, device=audio_embeds.device
            )
            emb = embed_tokens(ids).to(audio_embeds.dtype)   # (ts_len_i, dim)
            block_ts_embs.append(emb)
            ts_lens.append(emb.shape[0])

        # 3. Interleave: [audio_block | ts_embs] per block, then concatenate along seq dim
        segments = []
        for i in range(num_blocks):
            audio_block = audio_embeds[:, i * K:(i + 1) * K, :]          # (bsz, K, dim)
            ts_emb = block_ts_embs[i].unsqueeze(0).expand(bsz, -1, -1)   # (bsz, ts_len_i, dim)
            segments.append(torch.cat([audio_block, ts_emb], dim=1))     # (bsz, K+ts_len_i, dim)

        new_audio_embeds = torch.cat(segments, dim=1)   # (bsz, total_len, dim)

        # 4. Update valid lengths using cumulative sum of per-block timestamp token counts
        ts_lens_t = torch.tensor(ts_lens, dtype=torch.long, device=audio_embeds.device)
        ts_cumsum = torch.cumsum(ts_lens_t, dim=0)                         # (num_blocks,)
        num_valid_blocks = (audio_embeds_lens + K - 1) // K                # (bsz,)
        num_valid_blocks = (audio_embeds_lens + K - 1) // K
        num_valid_blocks = num_valid_blocks.clamp(1, num_blocks)
        new_audio_embeds_lens = audio_embeds_lens + ts_cumsum[num_valid_blocks - 1]

        return new_audio_embeds, new_audio_embeds_lens

    def expand_llm_vocab(self, tokenizer):
        """Add timestamp/speaker tokens to *tokenizer*.

        The embedding resize and grad-unfreezing are handled inside __init__,
        so this only updates the tokenizer. Call this only on the from-scratch
        path where the tokenizer starts from the unexpanded base LLM tokenizer.
        """
        added = tokenizer.add_special_tokens({"additional_special_tokens": self.NEW_SPECIAL_TOKENS})
        print(
            f"[SALMONN.expand_llm_vocab] Added {added} new special tokens; "
            f"LLM vocab size = {self.base_llm.config.vocab_size}."
        )
        return added

    # ------------------------------------------

    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        from models.Qformer import BertConfig, BertLMHeadModel
        encoder_config = BertConfig.from_pretrained(BERT_CKPT)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def _can_record_outputs(self):
        return getattr(self.base_llm, "_can_record_outputs", {})
    
    def get_audio_encoder(self, model_args):
        if model_args.encoder_type == "zipformer2":
            from spear_encoder.model import MultiKDModel
            from spear_encoder.scaling import ScheduledFloat
            from spear_encoder.subsampling import Conv2dSubsampling
            # from spear_encoder.zipformer import Zipformer2
            encoder_lora = getattr(model_args, "encoder_lora", False)
            if encoder_lora:
                from spear_encoder.zipformer_lora import Zipformer2
            else:
                from spear_encoder.zipformer_layerwise import Zipformer2
            def _to_int_tuple(s: str):
                return tuple(map(int, s.split(",")))

            encoder_embed = Conv2dSubsampling(
                in_channels=128,
                out_channels=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280")[0],
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            )
            if encoder_lora:
                encoder = Zipformer2(
                    output_downsampling_factor=1,
                    downsampling_factor=_to_int_tuple("1,2,4,8,4,2,1"),
                    num_encoder_layers=_to_int_tuple("1,2,3,4,1,1,1"),
                    encoder_dim=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280"),
                    encoder_unmasked_dim=_to_int_tuple("768,768,768,768,768,768,768"),
                    query_head_dim=_to_int_tuple("32"),
                    pos_head_dim=_to_int_tuple("4"),
                    value_head_dim=_to_int_tuple("12"),
                    pos_dim=48,
                    num_heads=_to_int_tuple("8,8,8,8,8,8,8"),
                    feedforward_dim=_to_int_tuple("3840,3840,3840,3840,3840,3840,3840"),
                    cnn_module_kernel=_to_int_tuple("31,31,15,15,15,31,31"),
                    dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                    warmup_batches=4000.0,
                    causal=False,
                    chunk_size=_to_int_tuple("-1"),
                    left_context_frames=_to_int_tuple("-1"),
                    use_lora=model_args.encoder_lora,
                    lora_r=model_args.encoder_lora_rank,
                )
                num_param = sum([p.numel() for p in encoder.parameters()])
                num_trainable = 0
                for name, p in encoder.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        p.requires_grad = True
                        num_trainable += p.numel()
                    else:
                        p.requires_grad = False

                print(
                    "A total of {} trainable parameters ({:.3f}% of the audio encoder)".format(
                        num_trainable, num_trainable / num_param * 100
                    )
                )
            else:
                encoder = Zipformer2(
                    output_downsampling_factor=1,
                    downsampling_factor=_to_int_tuple("1,2,4,8,4,2,1"),
                    num_encoder_layers=_to_int_tuple("1,2,3,4,1,1,1"),
                    encoder_dim=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280"),
                    encoder_unmasked_dim=_to_int_tuple("768,768,768,768,768,768,768"),
                    query_head_dim=_to_int_tuple("32"),
                    pos_head_dim=_to_int_tuple("4"),
                    value_head_dim=_to_int_tuple("12"),
                    pos_dim=48,
                    num_heads=_to_int_tuple("8,8,8,8,8,8,8"),
                    feedforward_dim=_to_int_tuple("3840,3840,3840,3840,3840,3840,3840"),
                    cnn_module_kernel=_to_int_tuple("31,31,15,15,15,31,31"),
                    dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                    warmup_batches=4000.0,
                    causal=False,
                    chunk_size=_to_int_tuple("-1"),
                    left_context_frames=_to_int_tuple("-1"),
                )

            audio_encoder = MultiKDModel(
                encoder_embed=encoder_embed,
                encoder=encoder,
                encoder_dim=max(_to_int_tuple("1280,1280,1280,1280,1280,1280,1280")),
                num_codebooks=0,
            )
        elif model_args.encoder_type == "spear_transformer":
            from spear_transformer_encoder.model import get_spear_transformer_encoder_600M
            audio_encoder = get_spear_transformer_encoder_600M()
            audio_encoder = audio_encoder.to(torch.bfloat16)
            
        elif model_args.encoder_type == "dasheng":
            from transformers import AutoModel

            audio_encoder = AutoModel.from_pretrained(
                model_args.audio_encoder_path,
                outputdim=None,
                trust_remote_code=True
            )
        
        elif self.encoder_type == "dasheng_wavlm":
            from transformers import AutoModel

            audio_encoder = AutoModel.from_pretrained(
                model_args.audio_encoder_path,
                outputdim=None,
                trust_remote_code=True
            )

            speech_encoder = AutoModel.from_pretrained(model_args.speech_encoder_path)
            return audio_encoder, speech_encoder
        elif self.encoder_type == "whisper_beats":
            from beats.modeling_whisper import WhisperModel
            from beats.BEATs import BEATsConfig, BEATs

            speech_encoder = WhisperModel.from_pretrained(model_args.speech_encoder_path).encoder
            beats_ckpt = torch.load(model_args.audio_encoder_path, map_location='cpu', weights_only=True)
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            audio_encoder = BEATs(beats_cfg)
            audio_encoder.load_state_dict(beats_ckpt['model'], assign=True) # modified

            return audio_encoder, speech_encoder
        elif self.encoder_type == "whisper":
            from beats.modeling_whisper import WhisperModel

            audio_encoder = WhisperModel.from_pretrained(model_args.audio_encoder_path).encoder

            return audio_encoder
        elif self.encoder_type == "qwenomni":
            from transformers import Qwen2_5OmniForConditionalGeneration

            audio_encoder = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_args.audio_encoder_path).thinker.audio_tower # modified
        elif self.encoder_type == "qwen3omni":
            from transformers import Qwen3OmniMoeForConditionalGeneration

            audio_encoder = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_args.audio_encoder_path).thinker.audio_tower
        elif self.encoder_type == "mimo":
            sys.path.append("/mnt/bn/audio-visual-llm-data6/wangsiyin/models/MiMo-Audio/src")
            from mimo_audio_tokenizer import MiMoAudioTokenizer

            audio_encoder = MiMoAudioTokenizer.from_pretrained(model_args.audio_encoder_path)
        elif self.encoder_type == "perception_av":
            from core.audio_visual_encoder import PEAudioVisual
            audio_encoder = PEAudioVisual.from_config(model_args.audio_encoder_path, pretrained=True).audio_visual_model.audio_model
        elif self.encoder_type == "audio_flamingo":
            from transformers import AudioFlamingo3ForConditionalGeneration
            audio_encoder = AudioFlamingo3ForConditionalGeneration.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/audio-flamingo-3-hf").audio_tower

        return audio_encoder

    def encode_audio(self, fbank_feature, fbank_feature_len, raw_wavs, freeze_encoder=None):
        if freeze_encoder is None:
            freeze_encoder = self._encoder_frozen
        with self.maybe_autocast(next(self.audio_encoder.parameters()).dtype):
            if self.encoder_type == "zipformer2":
                with torch.set_grad_enabled(not freeze_encoder):
                    audio_embeds, encoder_out_lens, middle_out = self.audio_encoder.forward_encoder(
                        fbank_feature[:,:max(fbank_feature_len),:],
                        fbank_feature_len
                    )
                if self.concat_encoder_features:
                    # assert not self.weighted_sum_encoder
                    # NOTE: maybe this layer norm is not necessary, but we keep it this way for now
                    middle_out = [F.layer_norm(m.permute(1,0,2), (m.shape[-1],)) for m in middle_out]
                    middle_out = torch.cat(middle_out, dim=-1) # (N,T,num_layers * C)
                    audio_embeds = self.concat_proj(middle_out) # (N,T,C)
                audio_embeds = self.ln_audio(audio_embeds)
            if self.encoder_type == "spear_transformer":
                with torch.set_grad_enabled(not freeze_encoder):
                    audio_embeds, encoder_out_lens, middle_out = self.audio_encoder.forward_encoder(
                        fbank_feature[:,:max(fbank_feature_len),:], 
                        fbank_feature_len
                    )
                if self.concat_encoder_features:
                    middle_out = [F.layer_norm(m, (m.shape[-1],)) for m in middle_out]
                    middle_out = torch.cat(middle_out, dim=-1) # (N,T,num_layers * C)
                    audio_embeds = self.concat_proj(middle_out) # (N,T,C)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "dasheng":
                with torch.set_grad_enabled(not freeze_encoder):
                    audio_embeds = self.audio_encoder(fbank_feature.transpose(1, 2)).hidden_states
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "dasheng_wavlm":
                with torch.set_grad_enabled(not freeze_encoder):
                    fbanks = self.fbank(fbank_feature, return_tensors="pt", device=fbank_feature.device).input_values
                    audio_embeds = self.audio_encoder(fbanks.to(torch.bfloat16)).hidden_states
                    speech_embeds = self.speech_encoder(fbank_feature).last_hidden_state
                seqlen_audio = audio_embeds.size(1)
                bsz, seqlen, ndim = speech_embeds.size()
                if seqlen_audio * 2 > seqlen:
                    num = seqlen_audio * 2 - seqlen
                    pad_embeds = torch.zeros((bsz, num, ndim), dtype=speech_embeds.dtype, device=speech_embeds.device)
                    speech_embeds = torch.cat((speech_embeds, pad_embeds), dim=1)
                    seqlen += num
                elif seqlen_audio * 2 < seqlen:
                    num = seqlen - seqlen_audio * 2
                    speech_embeds = speech_embeds[:, :-num, :]
                    seqlen -= num
                speech_embeds = speech_embeds.view(bsz, seqlen // 2, ndim * 2)
                audio_embeds = torch.cat([audio_embeds, speech_embeds], dim=-1)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "whisper_beats":
                with torch.set_grad_enabled(not freeze_encoder):
                    speech_embeds = self.speech_encoder(fbank_feature, return_dict=True).last_hidden_state
                    audio_embeds, _ = self.audio_encoder.extract_features(raw_wavs, padding_mask=torch.arange(raw_wavs.size(1)).unsqueeze(0).to(raw_wavs.device) >= fbank_feature_len.unsqueeze(1), feature_only=True)
                if audio_embeds.size(1) < speech_embeds.size(1):
                    audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                elif audio_embeds.size(1) > speech_embeds.size(1):
                    speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                audio_embeds = self.ln_audio(audio_embeds)
                speech_embeds = self.ln_speech(speech_embeds)
                audio_embeds = torch.cat([audio_embeds, speech_embeds], dim=-1)
            elif self.encoder_type == "whisper":
                with torch.set_grad_enabled(not freeze_encoder):
                    speech_embeds = self.audio_encoder(fbank_feature, return_dict=True).last_hidden_state
                audio_embeds = self.ln_audio(speech_embeds)
            elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
                fbank_feature = fbank_feature.permute(0, 2, 1)[raw_wavs.bool()].permute(1, 0)
                with torch.set_grad_enabled(not freeze_encoder):
                    if self.encoder_type == "qwenomni":
                        audio_feat_lengths, audio_output_lengths = self.audio_encoder._get_feat_extract_output_lengths(raw_wavs.sum(-1))
                        audio_outputs = self.audio_encoder(
                            fbank_feature,
                            feature_lens=raw_wavs.sum(-1),
                            aftercnn_lens=audio_feat_lengths,
                        )
                    elif self.encoder_type == "qwen3omni":
                        audio_output_lengths = self._get_feat_extract_output_lengths(raw_wavs.sum(-1))
                        audio_outputs = self.audio_encoder(
                            fbank_feature,
                            feature_lens=raw_wavs.sum(-1)
                        )
                audio_embeds = rnn.pad_sequence(torch.split(audio_outputs.last_hidden_state, audio_output_lengths.tolist(), dim=0),batch_first=True)            
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "mimo":
                with torch.set_grad_enabled(not freeze_encoder):
                    audio_embeds, _, encoder_output_length, _ = self.audio_encoder.encode(fbank_feature, fbank_feature_len, use_quantizer=False)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "perception_av":
                with torch.inference_mode():
                    encoder_outputs = self.audio_encoder(fbank_feature.unsqueeze(1), padding_mask=raw_wavs, input_features=None)
                audio_embeds = encoder_outputs.last_hidden_state
                encoder_output_length = encoder_outputs.audio_feature_padding_mask.sum(dim=-1)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "audio_flamingo":
                with torch.set_grad_enabled(not freeze_encoder):
                    encoder_outputs = self.audio_encoder(fbank_feature, input_features_mask=raw_wavs)
                audio_embeds = encoder_outputs.last_hidden_state
                audio_embeds = self.ln_audio(audio_embeds)

            if self.connector_type == "Qformer":
                assert self.encoder_type == "zipformer2"
                audio_embeds, audio_atts = self.forward_qformer(audio_embeds, audio_embeds_lens=encoder_out_lens)
                audio_embeds_len = torch.ceil(encoder_out_lens / 50) * 3
                audio_embeds_len = torch.clamp(audio_embeds_len, max=audio_embeds.shape[1])
                return audio_embeds, audio_embeds_len.to(torch.int64)
            
            # this is for MLP connnector
            elif self.connector_type == "MLP":
                bsz, seqlen, ndim = audio_embeds.size()
                if seqlen % self.connector_seg_size != 0:
                    pad_embeds = torch.zeros(
                        (bsz, (seqlen // self.connector_seg_size + 1) * self.connector_seg_size - seqlen, ndim), dtype=audio_embeds.dtype, device=audio_embeds.device
                    )
                    audio_embeds = torch.cat((audio_embeds, pad_embeds), dim=1)
                    bsz, seqlen, ndim = audio_embeds.size()
                
                audio_embeds = audio_embeds.view(bsz, seqlen // self.connector_seg_size, ndim * self.connector_seg_size)
                audio_embeds = self.connector(audio_embeds)

                if self.encoder_type == "zipformer2":
                    return audio_embeds, torch.ceil(encoder_out_lens/self.connector_seg_size).to(torch.int64)
                elif self.encoder_type == "spear_transformer":
                    return audio_embeds, torch.ceil(encoder_out_lens/self.connector_seg_size).to(torch.int64)
                elif self.encoder_type == "dasheng":
                    return audio_embeds, torch.ceil(fbank_feature_len/(4*self.connector_seg_size)).to(torch.int64)
                elif self.encoder_type == "dasheng_wavlm":
                    return audio_embeds, torch.ceil(fbank_feature_len/(640*self.connector_seg_size)).to(torch.int64)
                elif self.encoder_type == "whisper_beats" or self.encoder_type == "whisper":
                    return audio_embeds, [300] * bsz
                elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
                    return audio_embeds, torch.ceil(audio_output_lengths/self.connector_seg_size).to(torch.int64)
                elif self.encoder_type == "mimo":
                    return audio_embeds, torch.ceil(encoder_output_length/self.connector_seg_size).to(torch.int64)
                elif self.encoder_type == "perception_av":
                    return audio_embeds, torch.ceil(encoder_output_length/self.connector_seg_size).to(torch.int64)
                elif self.encoder_type == "audio_flamingo":
                    return audio_embeds, torch.ceil(fbank_feature_len/(4*self.connector_seg_size)).to(torch.int64)       
    
    def forward_qformer(self, audio_embeds, audio_embeds_lens=None):
        assert self.connector_type == "Qformer", "forward_qformer only works when connector_type is Qformer"
        with self.maybe_autocast():
            # skip pre-norm because we already have the layer norm    
            if audio_embeds_lens is not None:
                audio_atts = torch.arange(audio_embeds.size(1)).unsqueeze(0).to(audio_embeds.device) < audio_embeds_lens.unsqueeze(1)
            else:
                audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

            if True or self.window_level_Qformer: # TODO: currently we always use window-level Qformer, because we find that frame-level Qformer does not perform well, we will further investigate this issue in the future
                B, T, C = audio_embeds.shape
                # kernel = round(1500 * self.second_per_window / 30.0)
                kernel = round(self.second_per_window * 50) # TODO: make the frame-rate an attribute
                stride = round(self.second_stride * 50) # TODO: make the frame-rate an attribute
                kernel = (1, kernel)
                stride = (1, stride)
                audio_embeds_tr = audio_embeds.transpose(1, 2).unsqueeze(2)
                audio_embeds_overlap = F.unfold(audio_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                _, _, L = audio_embeds_overlap.shape
                audio_embeds_overlap = audio_embeds_overlap.view(B, -1, kernel[1], L)
                audio_embeds_overlap = torch.permute(audio_embeds_overlap, [0, 3, 2, 1])
                audio_embeds = audio_embeds_overlap.reshape(-1, kernel[1], C)
                audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long, device=audio_embeds.device)

            query_tokens = self.speech_query_tokens.expand(audio_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            audio_embeds = self.post_qformer_proj(query_output.last_hidden_state)

            if True or self.window_level_Qformer:
                audio_embeds = audio_embeds.view(B, -1, audio_embeds.size(2)).contiguous()

            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(audio_embeds.device)

        return audio_embeds, audio_atts
    
    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def prepare_inputs_labels_for_speech(self, input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, raw_wavs):
        if self.llm_type == "Qwen":
            bos_id = 151652
            padding_id = 151643
            eos_id = 151645 # <|im_end|>
        elif self.llm_type == "Llama":
            bos_id = 128002
            padding_id = 128004
        audio_embeds, audio_embeds_lens = self.encode_audio(fbank_feature, fbank_feature_len, raw_wavs)
        if self.inject_temporal_embedding:
            audio_embeds, audio_embeds_lens = self.inject_temporal_embeddings(audio_embeds, audio_embeds_lens)
        elif self.inject_temporal_embedding_nl:
            audio_embeds, audio_embeds_lens = self.inject_temporal_embeddings_with_natural_language(
                audio_embeds, audio_embeds_lens
            )
        _, seqlen, dim = audio_embeds.size()
        bsz = input_ids.shape[0]
        input_embeds_list = []
        attention_mask_list = []
        audio_num = 0
        if labels is not None:
            labels_list = []
            # For batched reasoning network: collect per-sample inputs during the loop,
            # run the network once after, then insert the results.
            reasoning_audio_list = []    # (A_i, hidden) per sample
            reasoning_text_list = []     # (T_i, hidden) per sample
            reasoning_insert_at = []     # int or None per sample
            for i in range(bsz):
                if is_peft_model(self.base_llm):
                    current_input_embeds = self.base_llm.model.model.embed_tokens(input_ids[i].to(torch.int64))
                else:
                    current_input_embeds = self.base_llm.model.embed_tokens(input_ids[i].to(torch.int64))
                current_labels = labels[i]
                # find where currect_input_ids == bos_id, return all the index, this will be used for injecting audio embedding
                bos_idx = (input_ids[i] == bos_id).nonzero(as_tuple=True)[0]
                eos_idx = (input_ids[i] == eos_id).nonzero(as_tuple=True)[0]
                last_bos_idx = bos_idx[-1].item()
                # the text prompt embedding should be between <audio> (which is the last bos) and the first <|im_end|>
                # +2 is to skip the <audio> token as well as the <|vision_end|>
                current_text_prompt_embed = current_input_embeds[last_bos_idx+2:eos_idx[0]]
                current_audio_embed_list = []
                # since the original audio can be chunked into multiple samples, we collect them
                # and concat them back to a single sample
                for idx in bos_idx:
                    audio_len = min(audio_embeds_lens[audio_num], seqlen)
                    current_audio_embed = audio_embeds[audio_num,:audio_len,:]
                    current_input_embeds = torch.cat(
                        (current_input_embeds[:idx+1],current_audio_embed,current_input_embeds[idx+1:]),
                        dim=0
                    )
                    current_labels = torch.cat((current_labels[:idx+1], torch.full((audio_len,), fill_value=-100, dtype=torch.long).to(current_labels.device),current_labels[idx+1:]),dim=0)
                    audio_num += 1
                    bos_idx += audio_len
                    current_audio_embed_list.append(current_audio_embed)
                current_audio_embed = torch.cat(current_audio_embed_list, dim=0)
                if self.num_pause_steps > 0:
                    first_response_positions = (current_labels != -100).nonzero(as_tuple=True)[0]
                    if len(first_response_positions) > 0:
                        insert_at = first_response_positions[0].item()
                        if self.use_reasoning_network:
                            # Defer insertion — collect inputs for the batched forward pass below
                            reasoning_audio_list.append(current_audio_embed)
                            reasoning_text_list.append(current_text_prompt_embed)
                            reasoning_insert_at.append(insert_at)
                        else:
                            if self.distinct_pause_embed:
                                pause_embeds = self.pause_embed.to(current_input_embeds.dtype)
                            else:
                                pause_embeds = self.pause_embed.to(current_input_embeds.dtype).expand(self.num_pause_steps, -1)
                            current_input_embeds = torch.cat([
                                current_input_embeds[:insert_at],
                                pause_embeds,
                                current_input_embeds[insert_at:]
                            ], dim=0)
                            pause_labels = torch.full(
                                (self.num_pause_steps,), -100, dtype=torch.long, device=current_labels.device
                            )
                            current_labels = torch.cat([
                                current_labels[:insert_at],
                                pause_labels,
                                current_labels[insert_at:]
                            ], dim=0)
                    elif self.use_reasoning_network:
                        reasoning_audio_list.append(None)
                        reasoning_text_list.append(None)
                        reasoning_insert_at.append(None)
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_labels.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
                labels_list.append(current_labels)

            # Batched reasoning network forward pass — runs once for the whole batch
            if self.use_reasoning_network and self.num_pause_steps > 0:
                valid_idx = [i for i, ins in enumerate(reasoning_insert_at) if ins is not None]
                if valid_idx:
                    audio_padded = rnn.pad_sequence(
                        [reasoning_audio_list[i] for i in valid_idx], batch_first=True
                    )  # (B', max_A, hidden)
                    text_padded = rnn.pad_sequence(
                        [reasoning_text_list[i] for i in valid_idx], batch_first=True
                    )  # (B', max_T, hidden)
                    device = audio_padded.device
                    audio_lens = [reasoning_audio_list[i].shape[0] for i in valid_idx]
                    text_lens  = [reasoning_text_list[i].shape[0]  for i in valid_idx]
                    # key_padding_mask: True = padding position to be ignored
                    audio_pad_mask = (
                        torch.arange(audio_padded.shape[1], device=device).unsqueeze(0)
                        >= torch.tensor(audio_lens, device=device).unsqueeze(1)
                    )
                    text_pad_mask = (
                        torch.arange(text_padded.shape[1], device=device).unsqueeze(0)
                        >= torch.tensor(text_lens, device=device).unsqueeze(1)
                    )
                    reasoning_out = self.reasoning_network(
                        text_padded, audio_padded,
                        text_key_padding_mask=text_pad_mask,
                        audio_key_padding_mask=audio_pad_mask,
                    )  # (B', N, d_model)
                    reasoning_embeds = self.post_reasoning_proj(reasoning_out)  # (B', N, hidden)

                    for k, i in enumerate(valid_idx):
                        insert_at = reasoning_insert_at[i]
                        pause_embeds = reasoning_embeds[k].to(input_embeds_list[i].dtype)  # (N, hidden)
                        input_embeds_list[i] = torch.cat([
                            input_embeds_list[i][:insert_at],
                            pause_embeds,
                            input_embeds_list[i][insert_at:]
                        ], dim=0)
                        pause_labels = torch.full(
                            (self.num_pause_steps,), -100, dtype=torch.long, device=labels_list[i].device
                        )
                        labels_list[i] = torch.cat([
                            labels_list[i][:insert_at],
                            pause_labels,
                            labels_list[i][insert_at:]
                        ], dim=0)
                        # Update attention mask to cover the inserted tokens
                        attention_mask_list[i] = torch.ones(
                            (len(input_embeds_list[i]),), dtype=torch.long, device=device
                        )
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            return input_embeds, attention_mask, labels
        else:
            reasoning_audio_list = []
            reasoning_text_list = []
            for i in range(bsz):
                if is_peft_model(self.base_llm):
                    current_input_embeds = self.base_llm.model.model.embed_tokens(input_ids[i].to(torch.int64))
                else:
                    current_input_embeds = self.base_llm.model.embed_tokens(input_ids[i].to(torch.int64))
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i] == bos_id).nonzero(as_tuple=True)[0]
                padding_idx = (input_ids[i] == padding_id).nonzero(as_tuple=True)[0]
                last_bos_idx = bos_idx[-1].item()
                if self.use_reasoning_network and self.num_pause_steps > 0:
                    eos_idx = (input_ids[i] == eos_id).nonzero(as_tuple=True)[0]
                    current_text_prompt_embed = current_input_embeds[last_bos_idx+2:eos_idx[0]]
                current_audio_embed_list = []
                for idx in bos_idx:
                    audio_len = min(audio_embeds_lens[audio_num], seqlen)
                    current_audio_embed = audio_embeds[audio_num,:audio_len,:]
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],current_audio_embed,current_input_embeds[idx+1:]),dim=0)
                    audio_num += 1
                    bos_idx += audio_len
                    padding_idx += audio_len
                    if self.use_reasoning_network and self.num_pause_steps > 0:
                        current_audio_embed_list.append(current_audio_embed)
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_input_embeds.device)
                current_attention_mask[padding_idx] = 0
                if self.num_pause_steps > 0:
                    if self.use_reasoning_network:
                        current_audio_embed = torch.cat(current_audio_embed_list, dim=0)
                        reasoning_audio_list.append(current_audio_embed)
                        reasoning_text_list.append(current_text_prompt_embed)
                    else:
                        if self.distinct_pause_embed:
                            pause_embeds = self.pause_embed.to(current_input_embeds.dtype)
                        else:
                            pause_embeds = self.pause_embed.to(current_input_embeds.dtype).expand(self.num_pause_steps, -1)
                        current_input_embeds = torch.cat([current_input_embeds, pause_embeds], dim=0)
                        pause_mask = torch.ones((self.num_pause_steps,), dtype=torch.long).to(current_input_embeds.device)
                        current_attention_mask = torch.cat([current_attention_mask, pause_mask], dim=0)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)

            # Batched reasoning network forward pass
            if self.use_reasoning_network and self.num_pause_steps > 0 and reasoning_audio_list:
                audio_padded = rnn.pad_sequence(reasoning_audio_list, batch_first=True)
                text_padded  = rnn.pad_sequence(reasoning_text_list,  batch_first=True)
                device = audio_padded.device
                audio_lens = [t.shape[0] for t in reasoning_audio_list]
                text_lens  = [t.shape[0] for t in reasoning_text_list]
                audio_pad_mask = (
                    torch.arange(audio_padded.shape[1], device=device).unsqueeze(0)
                    >= torch.tensor(audio_lens, device=device).unsqueeze(1)
                )
                text_pad_mask = (
                    torch.arange(text_padded.shape[1], device=device).unsqueeze(0)
                    >= torch.tensor(text_lens, device=device).unsqueeze(1)
                )
                reasoning_out = self.reasoning_network(text_padded, audio_padded,
                                       text_key_padding_mask=text_pad_mask,
                                       audio_key_padding_mask=audio_pad_mask)  # (B, N, d_model)
                reasoning_embeds = self.post_reasoning_proj(reasoning_out)      # (B, N, hidden)
                for i in range(bsz):
                    pause_embeds = reasoning_embeds[i].to(input_embeds_list[i].dtype)
                    input_embeds_list[i] = torch.cat([input_embeds_list[i], pause_embeds], dim=0)
                    pause_mask = torch.ones((self.num_pause_steps,), dtype=torch.long, device=device)
                    attention_mask_list[i] = torch.cat([attention_mask_list[i], pause_mask], dim=0)
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True, padding_side="left")
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0, padding_side="left")
            return input_embeds, attention_mask, None
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda",dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def forward(
        self, 
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        fbank_feature = None,
        fbank_feature_len = None,
        raw_wavs = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(f"Current rank: {int(os.environ.get('RANK', 0))}, fbank_feature_len: {fbank_feature_len}")
        input_embeds, attention_mask, labels = self.prepare_inputs_labels_for_speech(
            input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, raw_wavs
        )

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with self.maybe_autocast():
            outputs = self.base_llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return outputs
        
    def generate(
        self,
        input_ids,
        fbank_feature,
        fbank_feature_len,
        raw_wavs=None,
        max_new_tokens=0,
        logits_processor=[],
        generation_config=None,
        **kwargs,
    ):
        if generation_config is None:
            from transformers import GenerationConfig
            generation_config = GenerationConfig.from_model_config(self.config)

        # 使用 kwargs 更新 generation_config
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        if max_new_tokens == 0:
            max_new_tokens = generation_config.max_new_tokens
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        inputs_embeds, attention_mask, _ = self.prepare_inputs_labels_for_speech(
            input_ids, None, None, fbank_feature, fbank_feature_len, raw_wavs
        )

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[torch.tensor([generation_config.eos_token_id]).cuda()])])

        with self.maybe_autocast():
            outputs = self.base_llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=generation_config.num_beams,
                do_sample=generation_config.do_sample,
                min_length=generation_config.min_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                repetition_penalty=generation_config.repetition_penalty,
                length_penalty=generation_config.length_penalty,
                attention_mask=attention_mask,
                pad_token_id=self.base_llm.config.eos_token_id
            )
        
        return outputs
