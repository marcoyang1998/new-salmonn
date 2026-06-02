import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert left_paths_gemini JSONL into stage2-style SALMONN JSON."
    )
    parser.add_argument(
        "--input-jsonl",
        default="salmonn_data_v1.1/left_paths_gemini.json",
        help="Input left_paths_gemini JSONL path.",
    )
    parser.add_argument(
        "--output-json",
        default="salmonn_data_v1.1/stage2/left_paths_gemini_stage2.json",
        help="Output stage2-style JSON path.",
    )
    return parser.parse_args()


def convert_record(record: dict) -> dict:
    qa = record.get("QA", {}) or {}
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    audio_path = str(record.get("path", "")).strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": f"<audio>{question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "audios": [audio_path],
    }


def convert_file(input_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        dst.write('{\n  "data": [\n')

        first = True
        for line in src:
            if not line.strip():
                continue

            record = json.loads(line)
            converted = convert_record(record)

            if not first:
                dst.write(',\n')
            dst.write("    ")
            dst.write(json.dumps(converted, ensure_ascii=False, indent=2).replace("\n", "\n    "))

            first = False
            count += 1

        dst.write('\n  ]\n}\n')

    return count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_json)

    count = convert_file(input_path, output_path)

    print(f"input_file={input_path}")
    print(f"output_file={output_path}")
    print(f"converted_records={count}")


if __name__ == "__main__":
    main()