from argparse import ArgumentParser
from pathlib import Path

from inference_batch_test_set import TOKENIZER_PATH, main, str2bool


DEFAULT_RESULTS_FOLDER = "results_speaker_adaptation"


def parse_args():
    parser = ArgumentParser(description="Run speaker/accent adaptation ASR inference.")
    parser.add_argument("--test_set_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--results_folder", type=str, default=DEFAULT_RESULTS_FOLDER)
    parser.add_argument("--write_path_title", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="speaker_adaptation")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--disable_thinking", type=str2bool, default=True)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--split_audio", type=str2bool, default=False)
    parser.add_argument(
        "--task",
        type=str,
        default="speaker_adaptation",
        choices=["speaker_adaptation", "accent_adaptation"],
        help="Which adaptation task to run inference for.",
    )

    parser.add_argument("--use_beam_search", type=str2bool, default=False)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--use_nucleus_sampling", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_path_title = args.write_path_title or f"stage2_{args.model_id}_{args.task}"
    write_path = Path(args.results_folder) / f"{write_path_title}.jsonl"
    write_path.parent.mkdir(parents=True, exist_ok=True)

    main(
        args.test_set_path,
        write_path,
        args.batch_size,
        args.checkpoint_path,
        args.max_new_tokens,
        start=args.start,
        task_filter=[args.task],
        split_audio=args.split_audio,
        disable_thinking=args.disable_thinking,
        tokenizer_path=args.tokenizer_path,
        use_beam_search=args.use_beam_search,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        use_nucleus_sampling=args.use_nucleus_sampling,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        use_ctx_audio=True,
    )
