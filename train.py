import os
import sys
import json
import math
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
import logging
logging.basicConfig(level=logging.WARNING, force=True)

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback
from torch.utils.data import Sampler

from modeling_salmonn import SALMONN
from salmonn_datasets import SALMONN_Dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    base_llm_path: Optional[str] = field(default="")
    llm_type: str = field(default="Qwen")
    attn_implementation: str = field(default="flash_attention_2")
    lora: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    dora: bool = field(default=False)
    encoder_type: str = field(default="zipformer2")
    audio_encoder_path: str = field(default="/mnt/bn/audio-visual-llm-data6/ckpts/spear-encoder-streaming-600M-speech-only/spear-xlarge-non-streaming/iter-448000-avg-2.pt")
    speech_encoder_path: str = field(default="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm")
    freeze_encoder: bool = field(default=True)
    connector_type: str = field(default="MLP")
    connector_seg_size: int = field(default=5)
    connector_hid_size: int = field(default=4096)
    concat_encoder_features: bool = field(default=False)
    encoder_lora: bool = field(default=False)
    encoder_lora_rank: int = field(default=16)
    encoder_lora_alpha: int = field(default=16)
    encoder_lora_dropout: float = field(default=0.05)
    expand_vocab: bool = field(default=False)
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze all base_llm parameters. Incompatible with lora=True or expand_vocab=True."})
    inject_temporal_embedding: bool = field(default=False)
    inject_temporal_embedding_nl: bool = field(default=False)
    temporal_granularity: float = field(default=0.5, metadata={"help": "Timestamp injection granularity in seconds (e.g. 0.5 → <|0.50|> every 0.5 s)."})
    encoder_frame_rate: int = field(default=50, metadata={"help": "Audio encoder output frame rate in Hz before the connector (e.g. 50 for SPEAR/zipformer2)."})
    append_reason_embed_behind_audio: bool = field(default=False, metadata={"help": "Whether to append reasoning embeddings behind audio features."})
    use_reasoning_network: bool = field(default=False, metadata={"help": "Whether to insert a reasoning network between the audio encoder and LLM, taking the audio encoder output as input and producing new 'reasoning' tokens to insert into the LLM input. If False, reasoning_network is not used and num_pause_steps just controls how many <PAUSE> tokens are inserted with no additional reasoning features."})
    use_qwen3_embedding_model: bool = field(default=False, metadata={"help": "Whether to use Qwen3-Embedding hidden states instead of base LLM token embeddings for the reasoning network text input."})
    qwen3_embedding_model_path: str = field(default="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/Qwen3-Embedding-0.6B", metadata={"help": "Local path to the Qwen3 embedding model used when use_qwen3_embedding_model=True."})
    qwen3_embedding_tokenizer_path: str = field(default="", metadata={"help": "Optional local tokenizer path for the Qwen3 embedding model. Falls back to qwen3_embedding_model_path when empty."})
    qwen3_embedding_max_length: int = field(default=2048, metadata={"help": "Max token length for Qwen3 embedding tokenization."})
    freeze_qwen3_embedding_model: bool = field(default=True, metadata={"help": "Freeze the Qwen3 embedding model parameters when enabled."})
    reasoning_network_num_layers: int = field(default=5, metadata={"help": "Number of transformer layers in the reasoning network (if use_reasoning_network is True)."})
    reasoning_network_dim: int = field(default=1024, metadata={"help": "Hidden dimension of the reasoning network (if use_reasoning_network is True). Defaults to the LLM hidden size if not set."})
    num_pause_steps: int = field(default=0, metadata={"help": "Number of <PAUSE> tokens to insert before decoding"})
    distinct_pause_embed: bool = field(default=False, metadata={"help": "Whether to use distinct embeddings for each <PAUSE> token when num_pause_steps > 0. If False, all <PAUSE> tokens share the same embedding."})

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="")
    split_audio: bool = field(default=False)
    audio_chunk: int = field(default=60, metadata={"help": "Audio chunk size in seconds when split_audio is True."})
    shuffle_mc_options: bool = field(default=True)
    max_audio_duration: float = field(
        default=-1,
        metadata={"help": "Maximum audio duration in seconds. Entries with any audio longer than this are "
                          "removed. -1 (default) disables filtering."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    run_name: Optional[str] = field(default="debug")
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Keep collator-only dataset fields like user_prompt instead of pruning them before collation."},
    )
    group_by_audio_length: bool = field(
        default=False,
        metadata={"help": "Group training batches by audio duration to minimise intra-batch padding waste. "
                          "Requires the dataset JSON to have a 'durations' field (add via compute_audio_durations.py). "
                          "Batches are length-sorted then shuffled at the batch level each epoch."},
    )
    audio_length_bucket_boundaries: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Bucket boundary breakpoints in seconds for group_by_audio_length. "
                          "Samples are assigned to the bucket [boundaries[i], boundaries[i+1]). "
                          "Defaults to [0, 10, 20, 30, 40, 120] when not set."},
    )

    min_learning_rate: Optional[float] = field(default=None)
    from_scratch: bool = field(default=False)
    from_freeze_llm: bool = field(
        default=False,
        metadata={"help": "Load non-LLM weights from model_name_or_path and initialise "
                          "the LLM fresh from base_llm_path. Use when resuming from a "
                          "checkpoint where the LLM was frozen (no LoRA) into a stage "
                          "where LoRA is enabled."}
    )
    encoder_lr_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "LR multiplier for the audio encoder parameters (e.g. 0.1 = 1/10 of base LR). "
                          "Used when freeze_encoder=False, or after the encoder is unfrozen via encoder_unfreeze_step."},
    )
    encoder_unfreeze_step: int = field(
        default=0,
        metadata={"help": """
Controls encoder freezing behaviour (together with freeze_encoder):
  freeze_encoder=True,  encoder_unfreeze_step=0  -> encoder is always frozen.
  freeze_encoder=True,  encoder_unfreeze_step=N  -> encoder is frozen for the first N steps,
                                                    then unfrozen with lr = learning_rate * encoder_lr_ratio.
  freeze_encoder=False                           -> encoder is never frozen
                                                    (encoder_lr_ratio applies from step 0).
"""},
    )

def _load_non_llm_weights(model, checkpoint_path):
    """Load all weights from checkpoint except base_llm parameters."""
    import glob
    from safetensors.torch import load_file as safetensors_load

    shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
    use_safetensors = bool(shard_files)
    if not use_safetensors:
        shard_files = sorted(glob.glob(os.path.join(checkpoint_path, "*.bin")))

    non_llm_state = {}
    for shard_file in shard_files:
        shard = safetensors_load(shard_file) if use_safetensors else torch.load(shard_file, map_location="cpu", weights_only=True)
        for k, v in shard.items():
            if not k.startswith("base_llm"):
                non_llm_state[k] = v

    missing, unexpected = model.load_state_dict(non_llm_state, strict=False)
    non_llm_missing = [k for k in missing if not k.startswith("base_llm")]
    if non_llm_missing:
        print(f"[train] WARNING: {len(non_llm_missing)} non-LLM keys missing from checkpoint: {non_llm_missing}")
    print(f"[train] from_freeze_llm: loaded {len(non_llm_state)} non-LLM parameters from {checkpoint_path}")


def load_model_and_dataset(model_args, data_args, training_args):
    if training_args.from_scratch:
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_llm_path)
        model_config = AutoConfig.from_pretrained(os.path.join(model_args.base_llm_path,"config.json"))
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = model_args.attn_implementation
        model = SALMONN(model_config, model_args)
        if model_args.expand_vocab:
            model.expand_llm_vocab(tokenizer)
    elif training_args.from_freeze_llm:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(os.path.join(model_args.base_llm_path, "config.json"))
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = model_args.attn_implementation
        model = SALMONN(model_config, model_args)
        _load_non_llm_weights(model, model_args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json"))
        model = SALMONN.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            model_args=model_args,
            #attn_implementation=model_args.attn_implementation,
            torch_dtype="auto",
            low_cpu_mem_usage=False,
        )
    if model_args.use_reasoning_network and model_args.num_pause_steps > 0 and model_args.use_qwen3_embedding_model:
        model.init_qwen3_embedding_model()
    dataset = SALMONN_Dataset(data_args, tokenizer, model_args.encoder_type, model_args.llm_type)
    if model_args.inject_temporal_embedding:
        model.register_temporal_tokens(tokenizer)
    if getattr(model_args, "inject_temporal_embedding_nl", False):
        model.register_nl_timestamp_tokenizer(tokenizer)
    return tokenizer, model, dataset

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.remove_unused_columns:
        print("[train] Overriding remove_unused_columns=True -> False so collator-only fields like user_prompt are preserved.")
        training_args.remove_unused_columns = False

    config = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # Set environment variable for WANDB logging
    os.environ["WANDB_PROJECT"] = "salmonn_v1.1"

    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate

    tokenizer, model, dataset = load_model_and_dataset(model_args, data_args, training_args)

    # When doing delayed encoder unfreezing, encoder params must have requires_grad=True
    # from the very start so DeepSpeed ZeRO includes them in its partitioning.
    # Gradient flow is then controlled inside encode_audio via torch.set_grad_enabled,
    # toggled through model._encoder_frozen at the designated step.
    if training_args.encoder_unfreeze_step > 0:
        n = 0
        for name, param in model.named_parameters():
            if name.startswith("audio_encoder") or name.startswith("speech_encoder"):
                param.requires_grad_(True)
                n += 1
        # _encoder_frozen starts True; the callback flips it to False at unfreeze_step
        model._encoder_frozen = True
        print(
            f"[train] Enabled requires_grad on {n} encoder params for DeepSpeed ZeRO; "
            f"gradients suppressed via set_grad_enabled until step {training_args.encoder_unfreeze_step}."
        )

    class Collator:
        def __call__(self, samples):
            for i, s in enumerate(samples):
                assert "user_prompt" in s, (
                    f"Missing user_prompt in collator sample[{i}]. "
                    f"Available keys: {sorted(s.keys())}"
                )
                assert isinstance(s["user_prompt"], list), (
                    f"user_prompt in collator sample[{i}] must be a list, "
                    f"got {type(s['user_prompt']).__name__}"
                )
                assert len(s["user_prompt"]) == 1, (
                    f"user_prompt in collator sample[{i}] must contain exactly one item, "
                    f"got {len(s['user_prompt'])}"
                )
                assert isinstance(s["user_prompt"][0], str), (
                    f"user_prompt[0] in collator sample[{i}] must be a string, "
                    f"got {type(s['user_prompt'][0]).__name__}"
                )
            input_ids = pad_sequence([s["input_ids"] for s in samples], batch_first=True)
            attention_mask = pad_sequence([s["attention_mask"] for s in samples], batch_first=True)
            labels = pad_sequence([s["labels"] for s in samples], padding_value=-100, batch_first=True)
            fbank_feature = []
            fbank_feature_len = []
            raw_wav = []
            for s in samples:
                fbank_feature += s["fbank_feature"]
                fbank_feature_len += s["fbank_feature_len"]
                raw_wav += s["raw_wavs"]
            fbank_feature = pad_sequence(fbank_feature, batch_first=True)
            if raw_wav != []:
                raw_wav = pad_sequence(raw_wav, batch_first=True)
            fbank_feature_len = torch.tensor(fbank_feature_len)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "fbank_feature": fbank_feature,
                "fbank_feature_len": fbank_feature_len,
                "raw_wavs": raw_wav,
                "audio_files": [s["audio_files"] for s in samples],
                "user_prompts": [s["user_prompt"][0] for s in samples],
            }
        
    collator = Collator()

    class LRLoggingCallback(TrainerCallback):
        """Logs per-param-group learning rates every logging step."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.trainer is None or not hasattr(self.trainer, "optimizer") or self.trainer.optimizer is None:
                return
            for i, group in enumerate(self.trainer.optimizer.param_groups):
                key = f"lr_group_{i}"
                current_lr = group["lr"]
                if logs is not None:
                    logs[key] = current_lr
                print(f"  [LR] param_group[{i}]: {current_lr:.3e}")

        def set_trainer(self, trainer):
            self.trainer = trainer

    lr_callback = LRLoggingCallback()

    class EncoderUnfreezeCallback(TrainerCallback):
        """Unfreezes the audio encoder at a given global step by setting
        model._encoder_frozen = False, which causes encode_audio to switch
        from torch.set_grad_enabled(False) to True for encoder calls.
        Encoder params are already in the optimizer from step 0 for DeepSpeed ZeRO."""

        def __init__(self, unfreeze_step: int, encoder_lr_ratio: Optional[float]):
            self.unfreeze_step = unfreeze_step
            self.encoder_lr_ratio = encoder_lr_ratio if encoder_lr_ratio is not None else 1.0
            self._unfrozen = False

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self._unfrozen or state.global_step < self.unfreeze_step:
                return

            model._encoder_frozen = False

            encoder_lr = args.learning_rate * self.encoder_lr_ratio
            print(
                f"[EncoderUnfreezeCallback] Step {state.global_step}: "
                f"model._encoder_frozen set to False — encoder now updating, "
                f"lr={encoder_lr:.3e} (ratio={self.encoder_lr_ratio})"
            )
            self._unfrozen = True

    class AudioLengthGroupedSampler(Sampler):
        """Assigns samples to predefined duration buckets, shuffles within each
        bucket, then concatenates and returns ALL indices.  The HF Trainer /
        accelerate automatically stripes indices across ranks
        ([rank, rank+world_size, ...]), so every rank ends up with samples
        from the same bucket at each global step — no manual rank handling
        needed (and no risk of double-sharding from accelerate wrapping).

        Args:
            lengths:            per-sample max-audio-duration (seconds).
            batch_size:         per-GPU batch size.
            world_size:         total number of data-parallel ranks (used only
                                for padding so total is evenly divisible).
            seed:               base RNG seed (epoch is added on top).
            bucket_boundaries:  ascending list of duration breakpoints (secs).
                                Samples are assigned to bucket [b[i], b[i+1]).
                                Samples beyond the last boundary go into the
                                last bucket.
                                Default: [0, 10, 20, 30, 40, 120].
        """

        DEFAULT_BOUNDARIES = [0, 10, 20, 30, 40, 120]

        def __init__(self, lengths, batch_size, world_size=1, seed=0,
                     bucket_boundaries=None):
            self.lengths = lengths
            self.batch_size = batch_size
            self.world_size = world_size
            self.seed = seed
            self.bucket_boundaries = bucket_boundaries or self.DEFAULT_BOUNDARIES
            self.epoch = 0

            # Pad so total is divisible by (world_size * batch_size), ensuring
            # every rank sees exactly the same number of complete batches.
            n = len(lengths)
            global_batch = world_size * batch_size
            self.total_size = math.ceil(n / global_batch) * global_batch

        def set_epoch(self, epoch: int):
            self.epoch = epoch

        def __len__(self):
            return self.total_size

        def _get_bucket_id(self, duration: float) -> int:
            for i in range(len(self.bucket_boundaries) - 1):
                if duration < self.bucket_boundaries[i + 1]:
                    return i
            return len(self.bucket_boundaries) - 2

        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            n = len(self.lengths)
            n_buckets = len(self.bucket_boundaries) - 1

            # Assign each sample to a bucket
            buckets = [[] for _ in range(n_buckets)]
            for i, dur in enumerate(self.lengths):
                buckets[self._get_bucket_id(dur)].append(i)

            # Shuffle within each bucket, then concatenate
            shuffled = []
            for bucket in buckets:
                if not bucket:
                    continue
                perm = torch.randperm(len(bucket), generator=g).tolist()
                shuffled.extend(bucket[p] for p in perm)

            # Pad to total_size by repeating from the front
            shuffled = shuffled + shuffled[: self.total_size - n]

            # Shuffle the order of global batches (world_size * batch_size
            # consecutive indices) so the bucket ordering across training steps
            # is randomised, while keeping all ranks in the same bucket per step.
            global_batch = self.world_size * self.batch_size
            n_global_batches = self.total_size // global_batch
            global_batch_order = torch.randperm(n_global_batches, generator=g).tolist()

            indices = []
            for gb in global_batch_order:
                indices.extend(shuffled[gb * global_batch: (gb + 1) * global_batch])

            return iter(indices)

    class SALMONNTrainer(Trainer):
        """Trainer subclass that supports a separate (lower) LR for the audio encoder."""

        def create_optimizer(self):
            encoder_lr_ratio = self.args.encoder_lr_ratio
            use_delayed_unfreeze = self.args.encoder_unfreeze_step > 0
            split_encoder = (not model_args.freeze_encoder) or use_delayed_unfreeze

            if not split_encoder or encoder_lr_ratio == 1.0:
                return super().create_optimizer()

            base_lr = self.args.learning_rate
            encoder_lr = base_lr * encoder_lr_ratio

            encoder_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if name.startswith("audio_encoder") or name.startswith("speech_encoder"):
                    if use_delayed_unfreeze:
                        # Include encoder params regardless of requires_grad so DeepSpeed
                        # sets them up from the start (add_param_group is unsupported in ZeRO).
                        encoder_params.append(param)
                    elif param.requires_grad:
                        encoder_params.append(param)
                elif param.requires_grad:
                    other_params.append(param)

            param_groups = [
                {"params": other_params, "lr": base_lr},
                {"params": encoder_params, "lr": encoder_lr},
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            # Remove 'lr' from kwargs so our per-group LRs take effect
            optimizer_kwargs.pop("lr", None)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

            # Print a summary so it's easy to verify LRs at startup
            print(f"[SALMONNTrainer] Optimizer param groups:")
            print(f"  group[0] (other params):   {len(other_params):5d} tensors, lr={base_lr:.3e}")
            print(f"  group[1] (encoder params): {len(encoder_params):5d} tensors, lr={encoder_lr:.3e} {'(frozen until step ' + str(self.args.encoder_unfreeze_step) + ')' if use_delayed_unfreeze else ''}")
            return self.optimizer

        def _get_train_sampler(self, train_dataset=None):
            if train_dataset is None:
                train_dataset = self.train_dataset
            if not self.args.group_by_audio_length or not hasattr(train_dataset, "lengths"):
                return super()._get_train_sampler(train_dataset)

            world_size = int(os.environ.get("WORLD_SIZE", 1))
            sampler = AudioLengthGroupedSampler(
                lengths=train_dataset.lengths,
                batch_size=self.args.per_device_train_batch_size,
                world_size=world_size,
                seed=self.args.seed,
                bucket_boundaries=self.args.audio_length_bucket_boundaries,
            )
            print(
                f"[SALMONNTrainer] Using AudioLengthGroupedSampler "
                f"(world_size={world_size}, "
                f"batch_size={self.args.per_device_train_batch_size}, "
                f"n_samples={len(train_dataset)}, "
                f"boundaries={sampler.bucket_boundaries})"
            )
            return sampler

    # callbacks = [GPUStatsCallback()]
    callbacks = []
    if training_args.encoder_unfreeze_step > 0:
        if not model_args.freeze_encoder:
            raise ValueError(
                "encoder_unfreeze_step > 0 requires freeze_encoder=True. "
                "Modes: (freeze_encoder=True, encoder_unfreeze_step=0) = always frozen; "
                "(freeze_encoder=True, encoder_unfreeze_step=N) = frozen then unfrozen at step N; "
                "(freeze_encoder=False) = never frozen."
            )
        callbacks.append(
            EncoderUnfreezeCallback(
                unfreeze_step=training_args.encoder_unfreeze_step,
                encoder_lr_ratio=training_args.encoder_lr_ratio,
            )
        )
        print(
            f"[train] Encoder will be unfrozen at step {training_args.encoder_unfreeze_step} "
            f"with lr_ratio={training_args.encoder_lr_ratio}"
        )

    trainer = SALMONNTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer, # modified
        data_collator=collator,
        callbacks=callbacks,
    )
    # lr_callback.set_trainer(trainer)
    all_trainable = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_trainable.append(name)
    print(f"All trainable parameters: {all_trainable}")
    total_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    total_llm_lora_trainable = sum([p.numel() for name, p in model.named_parameters() if p.requires_grad and "lora" in name and "base_llm" in name])
    total_encoder_lora_trainable = sum([p.numel() for name, p in model.named_parameters() if p.requires_grad and "lora" in name and "encoder" in name])
    print(f"Total number of trainable parameters: {total_trainable}")
    print(f"Total number of trainable LLM LoRA parameters: {total_llm_lora_trainable}")
    print(f"Total number of trainable encoder LoRA parameters: {total_encoder_lora_trainable}")

    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from existing checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
