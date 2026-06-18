import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split a SALMONN-style JSON file into multiple JSON chunks.")
    parser.add_argument("--input", required=True, help="Input SALMONN-style JSON with top-level data list.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunk JSON files will be written.")
    parser.add_argument("--prefix", default="chunk", help="Chunk filename prefix.")
    parser.add_argument("--num-chunks", type=int, required=True, help="Number of chunks to create.")
    parser.add_argument("--progress-every", type=int, default=10000, help="Print progress after this many input items.")
    return parser.parse_args()


def iter_json_array_item_strings(path):
    chunk_size = 4 * 1024 * 1024
    found_array = False
    collecting = False
    in_string = False
    escape = False
    depth = 0
    item_parts = []

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if collecting:
                    raise ValueError(f"Unexpected EOF while reading an item from {path}")
                return

            pos = 0
            if not found_array:
                start = chunk.find("[")
                if start == -1:
                    continue
                found_array = True
                pos = start + 1

            for char in chunk[pos:]:
                if not collecting:
                    if char.isspace() or char == ",":
                        continue
                    if char == "]":
                        return
                    if char != "{":
                        raise ValueError(f"Expected object item in top-level data array, got {char!r}")
                    collecting = True
                    in_string = False
                    escape = False
                    depth = 1
                    item_parts = ["{"]
                    continue

                item_parts.append(char)
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        yield "".join(item_parts)
                        collecting = False
                        item_parts = []


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
        for index, item_text in enumerate(iter_json_array_item_strings(args.input)):
            rank = index % args.num_chunks
            if counts[rank] > 0:
                files[rank].write(",\n")
            files[rank].write(item_text)
            counts[rank] += 1
            if counts[rank] % 100 == 0:
                files[rank].flush()
            total = index + 1
            if args.progress_every > 0 and total % args.progress_every == 0:
                print(f"processed {total} items", flush=True)
        write_chunk_footer(files)
    finally:
        for f in files:
            f.close()

    for path, count in zip(paths, counts):
        print(f"{path}\t{count}")


if __name__ == "__main__":
    main()
