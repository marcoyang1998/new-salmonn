import os
import sys
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union
import logging
logging.basicConfig(level=logging.WARNING, force=True)

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoConfig, AutoTokenizer, Trainer

from modeling_salmonn import SALMONN
from datasets import SALMONN_Dataset

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
    encoder_type: str = field(default="zipformer2")
    audio_encoder_path: str = field(default="/mnt/bn/audio-visual-llm-data6/ckpts/spear-encoder-streaming-600M-speech-only/spear-xlarge-non-streaming/iter-448000-avg-2.pt")
    speech_encoder_path: str = field(default="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm")
    freeze_encoder: bool = field(default=True)
    connector_type: str = field(default="MLP")
    connector_seg_size: int = field(default=5)
    connector_hid_size: int = field(default=4096)

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default="")
    split_audio: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    run_name: Optional[str] = field(default="debug")

    min_learning_rate: Optional[float] = field(default=None)
    from_scratch: bool = field(default=False)

def load_model_and_dataset(model_args, data_args, training_args):
    if training_args.from_scratch:
        tokenizer = AutoTokenizer.from_pretrained(model_args.base_llm_path)
        model_config = AutoConfig.from_pretrained(os.path.join(model_args.base_llm_path,"config.json"))
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = model_args.attn_implementation
        model = SALMONN(model_config, model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json"))
        model = SALMONN.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            model_args=model_args,
            #attn_implementation=model_args.attn_implementation,
            torch_dtype="auto"
        )
    dataset = SALMONN_Dataset(data_args, tokenizer, model_args.encoder_type, model_args.llm_type)
    return tokenizer, model, dataset

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    class Collator:
        def __call__(self, samples):
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
            
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "fbank_feature": fbank_feature, "fbank_feature_len": fbank_feature_len, "raw_wavs": raw_wav}
        
    collator = Collator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer, # modified
        data_collator=collator
    )

    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
