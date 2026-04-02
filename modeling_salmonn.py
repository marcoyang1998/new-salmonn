from transformers import PreTrainedModel, AutoModelForCausalLM, StoppingCriteriaList, StoppingCriteria, AutoFeatureExtractor
from peft import LoraConfig, TaskType, get_peft_model
import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys

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

        
        if model_args.lora:
            for name, param in self.base_llm.named_parameters():
                param.requires_grad = False
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,    
                r=model_args.lora_rank,   # 64                          
                lora_alpha=model_args.lora_alpha,  # 64 
                lora_dropout=model_args.lora_dropout
            )
            self.base_llm = get_peft_model(self.base_llm, peft_config)
        
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
                self.audio_encoder.load_state_dict(torch.load(model_args.audio_encoder_path)["model"], strict=False)
        if model_args.freeze_encoder:
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            if self.encoder_type == "dasheng_wavlm" or self.encoder_type == "whisper_beats":
                for name, param in self.speech_encoder.named_parameters():
                    param.requires_grad = False
                self.speech_encoder.eval()
        if self.encoder_type == "zipformer2":
            encoder_dim = self.audio_encoder.encoder_dim
            num_encoder_layers = sum(self.audio_encoder.encoder.num_encoder_layers)
            if self.concat_encoder_features:
                # assert not self.weighted_sum_encoder, "Cannot concat encoder features when using weighted sum"
                num_encoder_layers = self.audio_encoder.num_encoder_layers
                self.concat_proj = nn.Linear(int(num_encoder_layers * encoder_dim), encoder_dim)
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

        if model_args.connector_type == "MLP":
            self.connector_seg_size = model_args.connector_seg_size
            self.connector_hid_size = model_args.connector_hid_size
            self.connector = nn.Sequential(
                nn.Linear(encoder_dim * self.connector_seg_size, self.connector_hid_size),
                nn.ReLU(),
                nn.Linear(self.connector_hid_size, config.hidden_size)
            )
    
    def _can_record_outputs(self):
        return getattr(self.base_llm, "_can_record_outputs", {})
    
    def get_audio_encoder(self, model_args):
        if model_args.encoder_type == "zipformer2":
            from spear_encoder.model import MultiKDModel
            from spear_encoder.scaling import ScheduledFloat
            from spear_encoder.subsampling import Conv2dSubsampling
            # from spear_encoder.zipformer import Zipformer2
            from spear_encoder.zipformer_layerwise import Zipformer2
            def _to_int_tuple(s: str):
                return tuple(map(int, s.split(",")))

            encoder_embed = Conv2dSubsampling(
                in_channels=128,
                out_channels=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280")[0],
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            )

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

    def encode_audio(self, fbank_feature, fbank_feature_len, raw_wavs):
        with self.maybe_autocast(next(self.audio_encoder.parameters()).dtype):
            if self.encoder_type == "zipformer2":
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
            elif self.encoder_type == "dasheng":
                audio_embeds = self.audio_encoder(fbank_feature.transpose(1, 2)).hidden_states
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "dasheng_wavlm":
                if self.freeze_encoder:
                    with torch.no_grad():
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
                speech_embeds = self.audio_encoder(fbank_feature, return_dict=True).last_hidden_state
                audio_embeds = self.ln_audio(speech_embeds)
            elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
                fbank_feature = fbank_feature.permute(0, 2, 1)[raw_wavs.bool()].permute(1, 0)
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
                audio_embeds, _, encoder_output_length, _ = self.audio_encoder.encode(fbank_feature, fbank_feature_len, use_quantizer=False)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "perception_av":
                with torch.inference_mode():
                    encoder_outputs = self.audio_encoder(fbank_feature.unsqueeze(1), padding_mask=raw_wavs, input_features=None)
                audio_embeds = encoder_outputs.last_hidden_state
                encoder_output_length = encoder_outputs.audio_feature_padding_mask.sum(dim=-1)
                audio_embeds = self.ln_audio(audio_embeds)
            elif self.encoder_type == "audio_flamingo":
                encoder_outputs = self.audio_encoder(fbank_feature, input_features_mask=raw_wavs)
                audio_embeds = encoder_outputs.last_hidden_state
                audio_embeds = self.ln_audio(audio_embeds)

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
    
    def _get_feat_extract_output_lengths(self, input_lengths):
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths

    def prepare_inputs_labels_for_speech(self, input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, raw_wavs):
        if self.llm_type == "Qwen":
            bos_id = 151652
            padding_id = 151643
        elif self.llm_type == "Llama":
            bos_id = 128002
            padding_id = 128004
        audio_embeds, audio_embeds_lens = self.encode_audio(fbank_feature, fbank_feature_len, raw_wavs)
        _, seqlen, dim = audio_embeds.size()
        bsz = input_ids.shape[0]
        input_embeds_list = []
        attention_mask_list = []
        audio_num = 0
        if labels is not None:
            labels_list = []
            for i in range(bsz):
                if is_peft_model(self.base_llm):
                    current_input_embeds = self.base_llm.model.model.embed_tokens(input_ids[i].to(torch.int64))
                else:
                    current_input_embeds = self.base_llm.model.embed_tokens(input_ids[i].to(torch.int64))
                current_labels = labels[i]
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i] == bos_id).nonzero(as_tuple=True)[0]
                for idx in bos_idx:
                    audio_len = min(audio_embeds_lens[audio_num], seqlen)
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],audio_embeds[audio_num,:audio_len,:],current_input_embeds[idx+1:]),dim=0)
                    current_labels = torch.cat((current_labels[:idx+1], torch.full((audio_len,), fill_value=-100, dtype=torch.long).to(current_labels.device),current_labels[idx+1:]),dim=0)
                    audio_num += 1
                    bos_idx += audio_len
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_labels.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
                labels_list.append(current_labels)
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            return input_embeds, attention_mask, labels
        else:
            for i in range(bsz):
                if is_peft_model(self.base_llm):
                    current_input_embeds = self.base_llm.model.model.embed_tokens(input_ids[i].to(torch.int64))
                else:
                    current_input_embeds = self.base_llm.model.embed_tokens(input_ids[i].to(torch.int64))
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i] == bos_id).nonzero(as_tuple=True)[0]
                padding_idx = (input_ids[i] == padding_id).nonzero(as_tuple=True)[0]
                for idx in bos_idx:
                    audio_len = min(audio_embeds_lens[audio_num], seqlen)
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],audio_embeds[audio_num,:audio_len,:],current_input_embeds[idx+1:]),dim=0)
                    audio_num += 1
                    bos_idx += audio_len
                    padding_idx += audio_len
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_input_embeds.device)
                current_attention_mask[padding_idx] = 0
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
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
