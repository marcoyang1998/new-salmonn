import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchaudio
from transformers import AutoConfig, AutoTokenizer, ClapConfig, ClapModel, ClapProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_salmonn import SALMONN
from Inference_scripts.inference_mmau import (
    ModelArguments,
    ZIPFORMER_LIKE,
    extract_features,
    get_fbank,
    parse_model_output,
    prepare_model_inputs,
    str2bool,
)
from Inference_scripts.inference_utils import maybe_init_qwen3_embedding_model, override_args_from_config


DEFAULT_CLAP_MODEL = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/larger_clap_general"


BASE_PROMPT_TEMPLATE = (
    "Identify the primary sound event in this audio clip.\n"
    "Choose exactly one label from the ESC-50 label set below and answer with only that label.\n"
    "Do not add any explanation, punctuation, or extra words.\n"
    "Use the label text exactly as written, including underscores when present.\n"
    "Use the audio as the primary evidence. The retrieved training statistics are only reference information.\n"
    "Labels:\n{labels}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval-augmented SALMONN sound classification on ESC-50.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--esc50_json",
        type=str,
        default="data/esc50_salmonn.json",
        help="Path to the ESC-50 SALMONN-format JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the annotated output JSON.",
    )
    parser.add_argument(
        "--rag_db_path",
        type=str,
        default=None,
        help="Optional path to save or reuse the retrieval database as an .npz file.",
    )
    parser.add_argument(
        "--train_folds",
        type=str,
        default="1,2,3,4",
        help="Comma-separated folds used to build the retrieval DB.",
    )
    parser.add_argument(
        "--test_folds",
        type=str,
        default="5",
        help="Comma-separated folds used for evaluation.",
    )
    parser.add_argument(
        "--cross_validate_5fold",
        action="store_true",
        help="Run 5-fold evaluation using each ESC-50 fold once as test and report averaged results.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved neighbors.")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        help="Audio encoder type. If not set, value from checkpoint config.json is used.",
    )
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument(
        "--rag_embedding_backend",
        type=str,
        choices=("salmonn", "clap"),
        default="salmonn",
        help="Embedding backend for the retrieval database and query retrieval.",
    )
    parser.add_argument(
        "--use_mlp_connector",
        type=str2bool,
        default=True,
        help="Whether to use the MLP connector output. Set to false to retrieve from the pre-connector SALMONN audio features.",
    )
    parser.add_argument(
        "--use_raw_encoder_features",
        type=str2bool,
        default=False,
        help="Whether to use model.encode_audio_raw_encoder(...) for SALMONN retrieval embeddings. Only valid when --rag_embedding_backend=salmonn.",
    )
    parser.add_argument(
        "--clap_model_name_or_path",
        type=str,
        default=DEFAULT_CLAP_MODEL,
        help="Path to the CLAP checkpoint used when --rag_embedding_backend=clap.",
    )
    parser.add_argument(
        "--clap_device",
        type=str,
        default="auto",
        help="Device for CLAP retrieval embeddings. Use 'auto', 'cpu', or a CUDA device like 'cuda:0'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of test samples to process.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum number of generated tokens.")
    return parser.parse_args()


def parse_fold_list(text):
    folds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise ValueError("At least one fold must be provided.")
    return sorted(set(folds))


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_label_list(payload):
    label_map = payload.get("label_map")
    if isinstance(label_map, dict) and label_map:
        return [label_map[str(idx)] for idx in sorted(int(key) for key in label_map.keys())]

    samples = payload.get("data", payload if isinstance(payload, list) else [])
    labels = []
    seen = set()
    for sample in samples:
        label = sample.get("label") or sample.get("category")
        if label is not None and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def build_base_prompt(labels):
    return BASE_PROMPT_TEMPLATE.format(labels="\n".join(labels))


def build_rag_context(retrieved_records, top_k):
    counts = {}
    best_similarity = {}
    for record in retrieved_records:
        label = record["category"]
        counts[label] = counts.get(label, 0) + 1
        best_similarity[label] = max(best_similarity.get(label, float("-inf")), record["similarity"])

    label_rows = sorted(
        counts.items(),
        key=lambda item: (-item[1], -best_similarity[item[0]], item[0]),
    )
    lines = [f"Retrieved reference from the training data (top {top_k} nearest samples by cosine similarity):"]
    for label, count in label_rows:
        lines.append(f"- {label}: {count}")
    return "\n".join(lines), label_rows


def build_prompt(base_prompt, rag_context):
    return f"{base_prompt}\n\n{rag_context}\n\nAnswer with only one ESC-50 label."


def get_feature_tensors(audio_path, fbank, model, model_args):
    feature, raw_wavs, audio_nums, split_feature_lens = extract_features(audio_path, fbank, model_args)
    enc = model_args.encoder_type

    if enc == "whisper_beats":
        feature_lens = [wav.size(0) for wav in raw_wavs]
    elif enc in ZIPFORMER_LIKE and model_args.split_audio and audio_nums:
        feature_lens = split_feature_lens
    else:
        feature_lens = [item.size(0) for item in feature]

    feature_t = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
    raw_wavs_t = (
        torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
        if raw_wavs
        else []
    )
    feature_lens_t = torch.tensor(feature_lens, device=model.device)
    return feature_t, feature_lens_t, raw_wavs_t, audio_nums


def encode_audio_path(audio_path, fbank, model, model_args, use_mlp_connector=True, use_raw_encoder_features=False):
    feature_t, feature_lens_t, raw_wavs_t, _ = get_feature_tensors(audio_path, fbank, model, model_args)

    with torch.inference_mode():
        if use_raw_encoder_features:
            audio_embeds, audio_embeds_lens = model.encode_audio_raw_encoder(
                feature_t,
                feature_lens_t,
            )
        else:
            audio_embeds, audio_embeds_lens = model.encode_audio(
                feature_t,
                feature_lens_t,
                raw_wavs_t,
                use_mlp_connector=use_mlp_connector,
            )

    if not torch.is_tensor(audio_embeds_lens):
        audio_embeds_lens = torch.tensor(audio_embeds_lens, device=audio_embeds.device)

    segments = []
    for chunk_idx in range(audio_embeds.size(0)):
        valid_len = int(audio_embeds_lens[chunk_idx].item())
        segments.append(audio_embeds[chunk_idx, :valid_len].detach().float().cpu())

    sequence = torch.cat(segments, dim=0)
    pooled_embedding = sequence.mean(dim=0)
    return pooled_embedding


def load_clap_model(model_name_or_path, device):
    try:
        model = ClapModel.from_pretrained(model_name_or_path)
    except ValueError as exc:
        checkpoint_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        if "upgrade torch to at least v2.6" not in str(exc) or not os.path.exists(checkpoint_path):
            raise
        config = ClapConfig.from_pretrained(model_name_or_path)
        model = ClapModel(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
    return model.to(device)


def load_audio_for_clap(audio_path, target_sample_rate):
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
    return audio.squeeze(0).numpy()


def encode_audio_path_clap(audio_path, processor, model, device, target_sample_rate):
    waveform = load_audio_for_clap(audio_path, target_sample_rate)
    inputs = processor(
        audio=waveform,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embedding = model.get_audio_features(**inputs)
    return embedding.detach().float().cpu().squeeze(0)


def load_model(model_name_or_path, encoder_type=None, concat_encoder_features=None):
    model_args = ModelArguments()
    model_args.model_name_or_path = model_name_or_path
    model_args = override_args_from_config(model_name_or_path, model_args)

    if encoder_type is not None:
        model_args.encoder_type = encoder_type
    if concat_encoder_features is not None:
        model_args.concat_encoder_features = concat_encoder_features

    if model_args.llm_type == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif model_args.llm_type == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct"
        )
    else:
        raise ValueError(f"Unsupported llm_type: {model_args.llm_type}")

    model = SALMONN.from_pretrained(
        model_name_or_path,
        config=AutoConfig.from_pretrained(os.path.join(model_name_or_path, "config.json")),
        model_args=model_args,
        torch_dtype="auto",
        device_map="auto",
    )
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()
    return model, tokenizer, model_args


def extract_prediction(raw_output, normalized_to_label):
    normalized_output = normalize_label(raw_output)
    if normalized_output in normalized_to_label:
        return normalized_to_label[normalized_output], "exact"

    matched_labels = [
        normalized_to_label[norm_label]
        for norm_label in normalized_to_label
        if norm_label and norm_label in normalized_output
    ]
    matched_labels = list(dict.fromkeys(matched_labels))
    if len(matched_labels) == 1:
        return matched_labels[0], "substring"

    close_matches = difflib.get_close_matches(
        normalized_output,
        list(normalized_to_label.keys()),
        n=1,
        cutoff=0.75,
    )
    if close_matches:
        return normalized_to_label[close_matches[0]], "fuzzy"

    return None, "unparsed"


def split_by_folds(samples, folds):
    folds_set = set(folds)
    return [sample for sample in samples if int(sample.get("fold", -1)) in folds_set]


def select_db_subset(db_embeddings, db_records, allowed_folds):
    allowed_folds_set = set(allowed_folds)
    keep_indices = [
        idx
        for idx, record in enumerate(db_records)
        if int(record["fold"]) in allowed_folds_set
    ]
    if not keep_indices:
        raise ValueError(f"No retrieval database entries found for folds {sorted(allowed_folds_set)}.")

    subset_embeddings = db_embeddings[keep_indices]
    subset_records = [db_records[idx] for idx in keep_indices]
    return subset_embeddings, subset_records


def build_rag_db(train_data, encode_embedding_fn):
    embeddings = []
    records = []
    total = len(train_data)

    for idx, sample in enumerate(train_data, start=1):
        audio_path = sample["audios"][0]
        pooled_embedding = encode_embedding_fn(audio_path)
        embeddings.append(pooled_embedding.numpy())
        records.append(
            {
                "filename": sample.get("filename", os.path.basename(audio_path)),
                "audio_path": audio_path,
                "category": sample.get("label", sample.get("category")),
                "target": int(sample.get("target", -1)),
                "fold": int(sample.get("fold", -1)),
            }
        )

        if idx % 100 == 0 or idx == total:
            print(f"[RAG DB] encoded {idx}/{total}")

    embeddings = np.stack(embeddings).astype(np.float32)
    return embeddings, records


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def summarize_metrics(total, parsed, correct, retrieval_top1_correct, retrieval_topk_contains_correct, retrieval_majority_correct):
    return {
        "num_test_samples": total,
        "num_parsed_predictions": parsed,
        "parsed_rate": parsed / total,
        "num_correct": correct,
        "accuracy": correct / total,
        "retrieval_top1_correct": retrieval_top1_correct,
        "retrieval_top1_accuracy": retrieval_top1_correct / total,
        "retrieval_topk_contains_correct": retrieval_topk_contains_correct,
        "retrieval_topk_contains_accuracy": retrieval_topk_contains_correct / total,
        "retrieval_majority_correct": retrieval_majority_correct,
        "retrieval_majority_accuracy": retrieval_majority_correct / total,
    }


def average_fold_metrics(fold_summaries):
    metric_keys = [
        "parsed_rate",
        "accuracy",
        "retrieval_top1_accuracy",
        "retrieval_topk_contains_accuracy",
        "retrieval_majority_accuracy",
    ]
    averages = {}
    for key in metric_keys:
        averages[f"avg_{key}"] = sum(summary[key] for summary in fold_summaries) / len(fold_summaries)
    return averages


def save_rag_db(
    rag_db_path,
    embeddings,
    records,
    db_folds,
    rag_embedding_backend,
    use_mlp_connector,
    use_raw_encoder_features,
    model_name_or_path,
    clap_model_name_or_path,
):
    path = Path(rag_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        filenames=np.asarray([record["filename"] for record in records]),
        audio_paths=np.asarray([record["audio_path"] for record in records]),
        categories=np.asarray([record["category"] for record in records]),
        targets=np.asarray([record["target"] for record in records]),
        folds=np.asarray([record["fold"] for record in records]),
        db_folds=np.asarray(db_folds),
        rag_embedding_backend=np.asarray([rag_embedding_backend]),
        use_mlp_connector=np.asarray([int(use_mlp_connector)], dtype=np.int64),
        use_raw_encoder_features=np.asarray([int(use_raw_encoder_features)], dtype=np.int64),
        model_name_or_path=np.asarray([model_name_or_path]),
        clap_model_name_or_path=np.asarray([clap_model_name_or_path]),
    )


def load_rag_db(
    rag_db_path,
    db_folds,
    rag_embedding_backend,
    use_mlp_connector,
    use_raw_encoder_features,
    model_name_or_path,
    clap_model_name_or_path,
):
    db = np.load(rag_db_path, allow_pickle=False)

    saved_db_folds = db["db_folds"].astype(np.int64).tolist()
    saved_rag_embedding_backend = str(db["rag_embedding_backend"][0])

    if saved_db_folds != list(db_folds):
        raise ValueError(
            f"RAG DB source folds {saved_db_folds} do not match requested source folds {list(db_folds)}."
        )
    if saved_rag_embedding_backend != rag_embedding_backend:
        raise ValueError(
            f"RAG DB backend {saved_rag_embedding_backend} does not match requested backend {rag_embedding_backend}."
        )

    if rag_embedding_backend == "salmonn":
        saved_use_mlp_connector = bool(int(db["use_mlp_connector"][0]))
        saved_use_raw_encoder_features = False
        if "use_raw_encoder_features" in db.files:
            saved_use_raw_encoder_features = bool(int(db["use_raw_encoder_features"][0]))
        saved_model_name_or_path = str(db["model_name_or_path"][0])
        if saved_use_raw_encoder_features != use_raw_encoder_features:
            raise ValueError(
                "RAG DB raw encoder feature setting "
                f"{saved_use_raw_encoder_features} does not match requested value {use_raw_encoder_features}."
            )
        if not use_raw_encoder_features and saved_use_mlp_connector != use_mlp_connector:
            raise ValueError(
                f"RAG DB use_mlp_connector={saved_use_mlp_connector} does not match requested value {use_mlp_connector}."
            )
        if saved_model_name_or_path != model_name_or_path:
            raise ValueError(
                f"RAG DB checkpoint {saved_model_name_or_path} does not match requested checkpoint {model_name_or_path}."
            )
    elif rag_embedding_backend == "clap":
        saved_clap_model_name_or_path = str(db["clap_model_name_or_path"][0])
        if saved_clap_model_name_or_path != clap_model_name_or_path:
            raise ValueError(
                f"RAG DB CLAP checkpoint {saved_clap_model_name_or_path} does not match requested checkpoint {clap_model_name_or_path}."
            )

    records = []
    for filename, audio_path, category, target, fold in zip(
        db["filenames"],
        db["audio_paths"],
        db["categories"],
        db["targets"],
        db["folds"],
    ):
        records.append(
            {
                "filename": str(filename),
                "audio_path": str(audio_path),
                "category": str(category),
                "target": int(target),
                "fold": int(fold),
            }
        )
    return db["embeddings"].astype(np.float32), records


def retrieve_top_k(query_embedding, db_embeddings, db_records, top_k):
    effective_top_k = min(top_k, len(db_records))
    query_norm = normalize_embeddings(query_embedding[None, :])[0]
    scores = db_embeddings @ query_norm
    top_indices = np.argsort(-scores)[:effective_top_k]

    retrieved = []
    for rank, db_idx in enumerate(top_indices, start=1):
        record = dict(db_records[int(db_idx)])
        record["similarity"] = float(scores[int(db_idx)])
        record["rank"] = rank
        retrieved.append(record)
    return retrieved


def get_retrieval_majority_label(rag_label_rows):
    if not rag_label_rows:
        return None
    return rag_label_rows[0][0]


def evaluate_fold(
    fold_name,
    test_data,
    retrieval_embeddings,
    retrieval_records,
    encode_embedding_fn,
    base_prompt,
    normalized_to_label,
    tokenizer,
    model,
    model_args,
    fbank,
    top_k,
    max_new_tokens,
):
    total = len(test_data)
    parsed = 0
    correct = 0
    retrieval_top1_correct = 0
    retrieval_topk_contains_correct = 0
    retrieval_majority_correct = 0

    annotated_test_data = []

    for idx, sample in enumerate(test_data, start=1):
        item = dict(sample)
        audio_path = item["audios"][0]
        query_embedding = encode_embedding_fn(audio_path).numpy().astype(np.float32)
        retrieved = retrieve_top_k(query_embedding, retrieval_embeddings, retrieval_records, top_k)
        rag_context, rag_label_rows = build_rag_context(retrieved, top_k=len(retrieved))
        prompt = build_prompt(base_prompt, rag_context)

        messages = [{"role": "user", "content": "<audio>" + prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        feature_t, feature_lens_t, raw_wavs_t, audio_nums = get_feature_tensors(audio_path, fbank, model, model_args)
        model_inputs = cast(dict[str, Any], prepare_model_inputs(text, audio_nums, tokenizer, model, model_args))

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                fbank_feature=feature_t,
                fbank_feature_len=feature_lens_t,
                raw_wavs=raw_wavs_t,
                user_prompts=[prompt],
                max_new_tokens=max_new_tokens,
            )

        raw_output = parse_model_output(generated_ids, tokenizer)
        predicted_label, parse_mode = extract_prediction(raw_output, normalized_to_label)
        target_label = item.get("label", item.get("category"))
        is_correct = predicted_label == target_label if predicted_label is not None else False
        retrieval_top1_label = retrieved[0]["category"] if retrieved else None
        retrieval_topk_labels = [record["category"] for record in retrieved]
        retrieval_majority_label = get_retrieval_majority_label(rag_label_rows)
        is_retrieval_top1_correct = retrieval_top1_label == target_label
        is_retrieval_topk_contains_correct = target_label in retrieval_topk_labels
        is_retrieval_majority_correct = retrieval_majority_label == target_label

        if predicted_label is not None:
            parsed += 1
        if is_correct:
            correct += 1
        if is_retrieval_top1_correct:
            retrieval_top1_correct += 1
        if is_retrieval_topk_contains_correct:
            retrieval_topk_contains_correct += 1
        if is_retrieval_majority_correct:
            retrieval_majority_correct += 1

        item["rag_prompt"] = prompt
        item["rag_context"] = rag_context
        item["rag_label_counts"] = [
            {"label": label, "count": count}
            for label, count in rag_label_rows
        ]
        item["rag_retrieved_examples"] = retrieved
        item["retrieval_top1_label"] = retrieval_top1_label
        item["retrieval_majority_label"] = retrieval_majority_label
        item["is_retrieval_top1_correct"] = is_retrieval_top1_correct
        item["is_retrieval_topk_contains_correct"] = is_retrieval_topk_contains_correct
        item["is_retrieval_majority_correct"] = is_retrieval_majority_correct
        item["model_output_raw"] = raw_output
        item["model_prediction"] = predicted_label
        item["prediction_parse_mode"] = parse_mode
        item["is_correct"] = is_correct
        annotated_test_data.append(item)

        if idx % 100 == 0 or idx == total:
            print(
                f"[{fold_name} {idx}/{total}] parsed: {parsed}/{idx} ({parsed / idx:.2%}), "
                f"accuracy: {correct}/{idx} ({correct / idx:.2%}), "
                f"retrieval top1: {retrieval_top1_correct}/{idx} ({retrieval_top1_correct / idx:.2%}), "
                f"retrieval majority: {retrieval_majority_correct}/{idx} ({retrieval_majority_correct / idx:.2%})"
            )

    summary = summarize_metrics(
        total=total,
        parsed=parsed,
        correct=correct,
        retrieval_top1_correct=retrieval_top1_correct,
        retrieval_topk_contains_correct=retrieval_topk_contains_correct,
        retrieval_majority_correct=retrieval_majority_correct,
    )
    return annotated_test_data, summary


def main():
    args = parse_args()
    all_folds = [1, 2, 3, 4, 5]
    train_folds = parse_fold_list(args.train_folds)
    test_folds = parse_fold_list(args.test_folds)
    if args.top_k <= 0:
        raise ValueError("--top_k must be a positive integer.")
    if args.use_raw_encoder_features and args.rag_embedding_backend != "salmonn":
        raise ValueError("--use_raw_encoder_features is only supported when --rag_embedding_backend=salmonn.")
    if args.cross_validate_5fold:
        train_folds = all_folds
        test_folds = all_folds

    with open(args.esc50_json, "r") as f:
        payload = json.load(f)

    all_samples = payload["data"] if isinstance(payload, dict) else payload
    if args.cross_validate_5fold:
        db_source_folds = all_folds
        db_source_data = split_by_folds(all_samples, db_source_folds)
        if not db_source_data:
            raise ValueError(f"No samples found for 5-fold source folds {db_source_folds}.")
    else:
        db_source_folds = train_folds
        db_source_data = split_by_folds(all_samples, db_source_folds)
        if not db_source_data:
            raise ValueError(f"No training samples found for folds {db_source_folds}.")

    model, tokenizer, model_args = load_model(
        args.model_name_or_path,
        encoder_type=args.encoder_type,
        concat_encoder_features=args.concat_encoder_features,
    )
    fbank = get_fbank(model_args)

    clap_processor = None
    clap_model = None
    clap_target_sample_rate = None
    clap_device = None
    if args.rag_embedding_backend == "clap":
        clap_device = resolve_device(args.clap_device)
        clap_processor = ClapProcessor.from_pretrained(args.clap_model_name_or_path)
        clap_model = load_clap_model(args.clap_model_name_or_path, clap_device)
        clap_model.eval()
        clap_target_sample_rate = clap_processor.feature_extractor.sampling_rate

    if args.rag_embedding_backend == "salmonn":
        encode_embedding_fn = lambda audio_path: encode_audio_path(
            audio_path,
            fbank,
            model,
            model_args,
            use_mlp_connector=args.use_mlp_connector,
            use_raw_encoder_features=args.use_raw_encoder_features,
        )
    else:
        encode_embedding_fn = lambda audio_path: encode_audio_path_clap(
            audio_path,
            clap_processor,
            clap_model,
            clap_device,
            clap_target_sample_rate,
        )

    if args.rag_db_path and os.path.exists(args.rag_db_path):
        print(f"Loading RAG DB from {args.rag_db_path}")
        db_embeddings, db_records = load_rag_db(
            args.rag_db_path,
            db_folds=db_source_folds,
            rag_embedding_backend=args.rag_embedding_backend,
            use_mlp_connector=args.use_mlp_connector,
            use_raw_encoder_features=args.use_raw_encoder_features,
            model_name_or_path=args.model_name_or_path,
            clap_model_name_or_path=args.clap_model_name_or_path,
        )
    else:
        print(f"Building RAG DB from folds {db_source_folds}")
        db_embeddings, db_records = build_rag_db(
            db_source_data,
            encode_embedding_fn=encode_embedding_fn,
        )
        if args.rag_db_path:
            save_rag_db(
                args.rag_db_path,
                db_embeddings,
                db_records,
                db_folds=db_source_folds,
                rag_embedding_backend=args.rag_embedding_backend,
                use_mlp_connector=args.use_mlp_connector,
                use_raw_encoder_features=args.use_raw_encoder_features,
                model_name_or_path=args.model_name_or_path,
                clap_model_name_or_path=args.clap_model_name_or_path,
            )
            print(f"Saved RAG DB to {args.rag_db_path}")

    db_embeddings = normalize_embeddings(db_embeddings)
    labels = get_label_list(payload if isinstance(payload, dict) else {"data": all_samples})
    base_prompt = build_base_prompt(labels)
    normalized_to_label = {normalize_label(label): label for label in labels}

    fold_evaluations = []
    all_annotated_test_data = []

    if args.cross_validate_5fold:
        evaluation_specs = [
            {
                "fold_name": f"fold-{test_fold}",
                "test_folds": [test_fold],
                "retrieval_folds": [fold for fold in all_folds if fold != test_fold],
            }
            for test_fold in all_folds
        ]
    else:
        evaluation_specs = [
            {
                "fold_name": "single-run",
                "test_folds": test_folds,
                "retrieval_folds": train_folds,
            }
        ]

    for spec in evaluation_specs:
        current_test_data = [dict(sample) for sample in split_by_folds(all_samples, spec["test_folds"])]
        if args.limit is not None:
            current_test_data = current_test_data[: args.limit]
        if not current_test_data:
            raise ValueError(f"No test samples found for folds {spec['test_folds']}.")

        retrieval_embeddings, retrieval_records = select_db_subset(
            db_embeddings,
            db_records,
            allowed_folds=spec["retrieval_folds"],
        )

        print(
            f"Evaluating {spec['fold_name']}: test folds {spec['test_folds']} "
            f"against retrieval folds {spec['retrieval_folds']}"
        )
        annotated_test_data, fold_summary = evaluate_fold(
            fold_name=spec["fold_name"],
            test_data=current_test_data,
            retrieval_embeddings=retrieval_embeddings,
            retrieval_records=retrieval_records,
            encode_embedding_fn=encode_embedding_fn,
            base_prompt=base_prompt,
            normalized_to_label=normalized_to_label,
            tokenizer=tokenizer,
            model=model,
            model_args=model_args,
            fbank=fbank,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
        )
        fold_summary["test_folds"] = spec["test_folds"]
        fold_summary["retrieval_folds"] = spec["retrieval_folds"]
        fold_evaluations.append(fold_summary)
        all_annotated_test_data.extend(annotated_test_data)

    overall_total = sum(summary["num_test_samples"] for summary in fold_evaluations)
    overall_parsed = sum(summary["num_parsed_predictions"] for summary in fold_evaluations)
    overall_correct = sum(summary["num_correct"] for summary in fold_evaluations)
    overall_retrieval_top1_correct = sum(summary["retrieval_top1_correct"] for summary in fold_evaluations)
    overall_retrieval_topk_contains_correct = sum(summary["retrieval_topk_contains_correct"] for summary in fold_evaluations)
    overall_retrieval_majority_correct = sum(summary["retrieval_majority_correct"] for summary in fold_evaluations)
    overall_summary = summarize_metrics(
        total=overall_total,
        parsed=overall_parsed,
        correct=overall_correct,
        retrieval_top1_correct=overall_retrieval_top1_correct,
        retrieval_topk_contains_correct=overall_retrieval_topk_contains_correct,
        retrieval_majority_correct=overall_retrieval_majority_correct,
    )
    average_summary = average_fold_metrics(fold_evaluations) if args.cross_validate_5fold else None

    output_payload = {}
    if isinstance(payload, dict):
        output_payload.update({key: value for key, value in payload.items() if key != "data"})
    output_payload["data"] = all_annotated_test_data
    output_payload["rag_config"] = {
        "checkpoint": args.model_name_or_path,
        "encoder_type": model_args.encoder_type,
        "cross_validate_5fold": args.cross_validate_5fold,
        "train_folds": train_folds if not args.cross_validate_5fold else None,
        "test_folds": test_folds if not args.cross_validate_5fold else all_folds,
        "rag_db_source_folds": db_source_folds,
        "top_k": args.top_k,
        "rag_embedding_backend": args.rag_embedding_backend,
        "use_mlp_connector": args.use_mlp_connector,
        "use_raw_encoder_features": args.use_raw_encoder_features if args.rag_embedding_backend == "salmonn" else None,
        "clap_model_name_or_path": args.clap_model_name_or_path if args.rag_embedding_backend == "clap" else None,
        "rag_db_path": args.rag_db_path,
    }
    output_payload["base_prompt"] = base_prompt
    output_payload["fold_summaries"] = fold_evaluations
    output_payload["inference_summary"] = {
        "num_db_source_samples": len(db_source_data),
        "max_new_tokens": args.max_new_tokens,
        **overall_summary,
    }
    if average_summary is not None:
        output_payload["cross_validation_average"] = average_summary

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n===== ESC-50 RAG Inference Summary =====")
    print(f"Checkpoint : {args.model_name_or_path}")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"5-fold CV  : {args.cross_validate_5fold}")
    print(f"DB folds   : {db_source_folds}")
    print(f"Top-k      : {args.top_k}")
    print(f"RAG embed  : {args.rag_embedding_backend}")
    print(f"Raw enc    : {args.use_raw_encoder_features if args.rag_embedding_backend == 'salmonn' else 'n/a'}")
    print(f"MLP conn   : {args.use_mlp_connector if args.rag_embedding_backend == 'salmonn' else 'n/a'}")
    print(f"Eval size  : {overall_total}")
    print(f"Ret top1   : {overall_retrieval_top1_correct} ({overall_summary['retrieval_top1_accuracy']:.2%})")
    print(f"Ret in topk: {overall_retrieval_topk_contains_correct} ({overall_summary['retrieval_topk_contains_accuracy']:.2%})")
    print(f"Ret majority: {overall_retrieval_majority_correct} ({overall_summary['retrieval_majority_accuracy']:.2%})")
    print(f"Parsed     : {overall_parsed} ({overall_summary['parsed_rate']:.2%})")
    print(f"Correct    : {overall_correct} ({overall_summary['accuracy']:.2%})")
    if average_summary is not None:
        print("----- 5-fold average -----")
        print(f"Avg parsed     : {average_summary['avg_parsed_rate']:.2%}")
        print(f"Avg accuracy   : {average_summary['avg_accuracy']:.2%}")
        print(f"Avg ret top1   : {average_summary['avg_retrieval_top1_accuracy']:.2%}")
        print(f"Avg ret in topk: {average_summary['avg_retrieval_topk_contains_accuracy']:.2%}")
        print(f"Avg ret majority: {average_summary['avg_retrieval_majority_accuracy']:.2%}")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()
