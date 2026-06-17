import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split a SALMONN-style JSON file into multiple JSON chunks.")
    parser.add_argument("--input", required=True, help="Input SALMONN-style JSON with top-level data list.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunk JSON files will be written.")
    parser.add_argument("--prefix", default="chunk", help="Chunk filename prefix.")
    parser.add_argument("--num-chunks", type=int, required=True, help="Number of chunks to create.")
    return parser.parse_args()


def iter_json_array_items(path):
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = ""
    found_array = False

    with open(path, "r", encoding="utf-8") as f:
        while not found_array:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find top-level data array in {path}")
            buffer += chunk
            start = buffer.find("[")
            if start != -1:
                buffer = buffer[start + 1 :]
                found_array = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue

            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    yield item
                    buffer = buffer[end:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk


def write_chunk_header(files):
    for f in files:
        f.write('{"data": [\n')


def write_chunk_footer(files):
    for f in files:
        f.write("\n]}\n")


def main():
    args = parse_args()
    if args.num_chunks < 1:
        raise ValueError("--num-chunks must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(args.num_chunks - 1)))
    paths = [
        output_dir / f"{args.prefix}_{rank:0{width}d}.json"
        for rank in range(args.num_chunks)
    ]
    files = [path.open("w", encoding="utf-8") for path in paths]
    counts = [0] * args.num_chunks

    try:
        write_chunk_header(files)
        for index, item in enumerate(iter_json_array_items(args.input)):
            rank = index % args.num_chunks
            if counts[rank] > 0:
                files[rank].write(",\n")
            files[rank].write(json.dumps(item, ensure_ascii=False, separators=(", ", ": ")))
            counts[rank] += 1
        write_chunk_footer(files)
    finally:
        for f in files:
            f.close()

    for path, count in zip(paths, counts):
        print(f"{path}\t{count}")


if __name__ == "__main__":
    main()
