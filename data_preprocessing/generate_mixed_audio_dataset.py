"""
Generate a randomly mixed audio dataset from a JSON dataset file.

Each entry in the input JSON has:
  - "audios": list of audio file paths (one per entry)
  - "messages": conversation list; the assistant turn contains the speech transcript

For each mixture:
  - Randomly sample 2 or 3 entries from the dataset
  - Mix their audio with random delays (each source i > 0 gets a random delay
    sampled uniformly in [0, max_delay_seconds])
  - Record start_time / end_time / transcript for every source in the mixture

Output:
  - Mixed wav files saved under <output_dir>/mixed_audio/
  - A JSON metadata file with per-mixture source timing and transcript info
  - Optionally, a new dataset JSON in the same format as the input (for training)

Usage:
  python generate_mixed_audio_dataset.py \
      --input_json /path/to/dataset.json \
      --output_dir /path/to/output \
      --num_mixtures 10000 \
      --max_delay_seconds 2.0 \
      --sample_rate 16000 \
      --num_workers 8
"""

import os
import json
import random
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-10


# ---------------------------------------------------------------------------
# Helper: extract transcript from a messages list
# ---------------------------------------------------------------------------

def extract_transcript(messages):
    """Return the assistant's reply (the ASR transcript)."""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

def load_audio(path, target_sr=16000):
    """Load audio, resample to target_sr, return mono float32 numpy array."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as e:
        raise RuntimeError(f"Cannot read {path}: {e}")

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // g, sr // g)

    return audio.astype(np.float32)


def save_audio(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


# ---------------------------------------------------------------------------
# Mixing logic
# ---------------------------------------------------------------------------

def mix_sources_with_delays(sources, delay_samples_list):
    """
    Mix a list of mono audio arrays, each preceded by its delay.

    Args:
        sources            : list of np.ndarray, each shape (T_i,)
        delay_samples_list : list of int, delay in samples before each source

    Returns:
        mixture         : np.ndarray of the mixed signal
        start_samples   : list of int (absolute start sample for each source)
        end_samples     : list of int (absolute end sample for each source)
    """
    assert len(sources) == len(delay_samples_list)

    start_samples = delay_samples_list[:]
    end_samples = [d + len(s) for d, s in zip(delay_samples_list, sources)]
    total_len = max(end_samples)

    mixture = np.zeros(total_len, dtype=np.float32)
    for src, start, end in zip(sources, start_samples, end_samples):
        mixture[start:end] += src

    return mixture, start_samples, end_samples


def rms(signal):
    """Root-mean-square energy of a signal."""
    return np.sqrt(np.mean(signal ** 2) + EPS)


def sample_gains(sources, snr_range_db=(-5.0, 5.0)):
    """
    Return a gain vector so that every secondary source is within
    snr_range_db of the first source's RMS level.

    The first source is always kept at gain=1.0.  For each subsequent
    source a random SNR offset (in dB) is drawn uniformly from
    snr_range_db, and the gain that achieves that offset relative to
    source 0 is computed.

    Args:
        sources      : list of np.ndarray mono signals
        snr_range_db : (min_db, max_db) relative to source 0

    Returns:
        gains : list of float, one per source
    """
    ref_rms = rms(sources[0])
    gains = [1.0]
    for src in sources[1:]:
        snr_db = random.uniform(*snr_range_db)
        # target_rms / src_rms = 10^(snr_db/20)
        # where target_rms = ref_rms * 10^(snr_db/20)
        target_rms = ref_rms * (10 ** (snr_db / 20.0))
        src_rms = rms(src)
        gains.append(float(target_rms / src_rms))
    return gains


def normalize_mixture(mixture):
    """Peak-normalize to [-1, 1] to avoid clipping."""
    peak = np.abs(mixture).max()
    if peak > EPS:
        mixture = mixture / peak * 0.9
    return mixture


# ---------------------------------------------------------------------------
# Worker function (used in parallel processing)
# ---------------------------------------------------------------------------

def _mix_one(args):
    """
    Build one mixture.  Called in a worker process.

    args: (mix_idx, sampled_items, output_audio_dir, sample_rate, max_delay_seconds)

    Returns a dict with mixture metadata, or None on failure.
    """
    mix_idx, sampled_items, output_audio_dir, sample_rate, max_delay_seconds = args

    try:
        # Load audio for each source
        sources = []
        for item in sampled_items:
            audio_path = item["audios"][0]
            audio = load_audio(audio_path, target_sr=sample_rate)
            sources.append(audio)

        n_src = len(sources)

        # Sample gains: source 0 is reference; secondary sources within ±5 dB
        gains = sample_gains(sources, snr_range_db=(-5.0, 5.0))
        sources = [src * g for src, g in zip(sources, gains)]

        # Sample delays: source 0 always starts at 0; sources 1..n get random delay
        delay_samples_list = [0]
        for _ in range(1, n_src):
            delay_sec = random.uniform(0, max_delay_seconds)
            delay_samples_list.append(int(delay_sec * sample_rate))

        # Mix
        mixture, start_samples, end_samples = mix_sources_with_delays(
            sources, delay_samples_list
        )
        mixture = normalize_mixture(mixture)

        # Save mixture
        mix_filename = f"mix_{mix_idx:08d}.wav"
        mix_path = os.path.join(output_audio_dir, mix_filename)
        save_audio(mix_path, mixture, sample_rate)

        # Build metadata record
        record = {
            "mixture_path": mix_path,
            "mixture_duration": len(mixture) / sample_rate,
            "n_sources": n_src,
            "sources": [],
        }
        for i, (item, start, end, gain) in enumerate(
            zip(sampled_items, start_samples, end_samples, gains)
        ):
            record["sources"].append(
                {
                    "source_index": i + 1,
                    "audio_path": item["audios"][0],
                    "transcript": extract_transcript(item.get("messages", [])),
                    "start_time": round(start / sample_rate, 6),
                    "end_time": round(end / sample_rate, 6),
                    "gain": round(gain, 6),
                    "snr_db_vs_source1": round(20 * np.log10(gain * rms(sources[i]) / (rms(sources[0]) + EPS) + EPS), 2) if i > 0 else 0.0,
                }
            )

        return record

    except Exception as e:
        logger.warning(f"[mix {mix_idx}] Failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Build training-format JSON entries from metadata
# ---------------------------------------------------------------------------

def build_training_entries(metadata_records):
    """
    Convert mixture metadata into the same JSON schema as the input dataset.

    The user content prompt asks for overlapped speech transcription;
    the assistant content lists each speaker's transcript with timestamps.
    """
    entries = []
    for rec in metadata_records:
        sources = rec["sources"]

        # Build a combined transcript description
        source_lines = []
        for src in sources:
            source_lines.append(
                f"Speaker {src['source_index']} "
                f"[{src['start_time']:.2f}s - {src['end_time']:.2f}s]: "
                f"{src['transcript']}"
            )
        combined = "\n".join(source_lines)

        entry = {
            "audios": [rec["mixture_path"]],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "<audio>This audio contains overlapping speech from "
                        f"{rec['n_sources']} speakers. "
                        "Please transcribe each speaker's speech with start and end timestamps."
                    ),
                },
                {
                    "role": "assistant",
                    "content": combined,
                },
            ],
            "task_type": "mixed_asr",
            "mixture_metadata": rec,
        }
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a randomly mixed audio dataset."
    )
    p.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON dataset file.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for output files.",
    )
    p.add_argument(
        "--num_mixtures",
        type=int,
        default=10000,
        help="Number of mixtures to generate.",
    )
    p.add_argument(
        "--max_delay_seconds",
        type=float,
        default=2.0,
        help="Maximum delay in seconds between sources (default: 2.0).",
    )
    p.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for all audio (default: 16000).",
    )
    p.add_argument(
        "--min_sources",
        type=int,
        default=2,
        choices=[2, 3],
        help="Minimum number of sources per mixture (2 or 3, default: 2).",
    )
    p.add_argument(
        "--max_sources",
        type=int,
        default=3,
        choices=[2, 3],
        help="Maximum number of sources per mixture (2 or 3, default: 3).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--save_training_json",
        action="store_true",
        help="Also save a training-format JSON file alongside the metadata.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load input dataset
    logger.info(f"Loading dataset from {args.input_json} ...")
    with open(args.input_json, "r") as f:
        dataset = json.load(f)

    # Support both {"data": [...]} and plain list formats
    if isinstance(dataset, dict) and "data" in dataset:
        items = dataset["data"]
    elif isinstance(dataset, list):
        items = dataset
    else:
        raise ValueError("Unrecognised JSON format: expected a list or {'data': [...]}")

    logger.info(f"Loaded {len(items)} items.")
    if len(items) < args.min_sources:
        raise ValueError("Dataset too small to form any mixture.")

    # Prepare output dirs
    output_audio_dir = os.path.join(args.output_dir, "mixed_audio")
    os.makedirs(output_audio_dir, exist_ok=True)

    # Sample groups of 2 or 3 items for each mixture
    logger.info(
        f"Sampling {args.num_mixtures} mixtures "
        f"(n_sources in [{args.min_sources}, {args.max_sources}]) ..."
    )
    mixture_groups = []
    for _ in range(args.num_mixtures):
        n_src = random.randint(args.min_sources, args.max_sources)
        n_src = min(n_src, len(items))
        sampled = random.sample(items, k=n_src)
        mixture_groups.append(sampled)

    # Build worker args
    worker_args = [
        (
            mix_idx,
            sampled_items,
            output_audio_dir,
            args.sample_rate,
            args.max_delay_seconds,
        )
        for mix_idx, sampled_items in enumerate(mixture_groups)
    ]

    # Run mixing in parallel
    metadata_records = []
    logger.info(f"Mixing audio with {args.num_workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_mix_one, a): a[0] for a in worker_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Mixing"):
            result = future.result()
            if result is not None:
                metadata_records.append(result)

    # Sort by mix index for deterministic ordering
    metadata_records.sort(key=lambda r: r["mixture_path"])

    logger.info(
        f"Generated {len(metadata_records)} mixtures "
        f"({args.num_mixtures - len(metadata_records)} failed)."
    )

    # Save metadata JSON
    metadata_path = os.path.join(args.output_dir, "mixed_dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {metadata_path}")

    # Optionally save training-format JSON
    if args.save_training_json:
        training_entries = build_training_entries(metadata_records)
        training_path = os.path.join(args.output_dir, "mixed_dataset_train.json")
        with open(training_path, "w") as f:
            json.dump({"data": training_entries}, f, indent=2, ensure_ascii=False)
        logger.info(f"Training JSON saved to {training_path}")


if __name__ == "__main__":
    main()
