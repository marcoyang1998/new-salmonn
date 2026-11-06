"""Memory-bounded readers for VoxParadox JSONL and JSON-array manifests."""
import json

DATASET_ROOT = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/VoxParadox"


def iter_records(path):
    """Yield one record at a time; JSON arrays use a 64 KiB incremental decoder."""
    with open(path, encoding="utf-8") as stream:
        first = ""
        while not first:
            char = stream.read(1)
            if not char:
                return
            first = char.strip()
        stream.seek(0)
        if first != "[":
            for line in stream:
                if line.strip():
                    yield json.loads(line)
            return
        decoder = json.JSONDecoder()
        buffer = ""
        eof = False

        def refill():
            nonlocal buffer, eof
            chunk = stream.read(65536)
            buffer += chunk
            eof = not chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                refill()
                buffer = buffer.lstrip()

        whitespace()
        buffer = buffer[1:]
        whitespace()
        if buffer.startswith("]"):
            buffer = buffer[1:]
        else:
            while True:
                while True:
                    try:
                        item, end = decoder.raw_decode(buffer)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        refill()
                if not isinstance(item, dict):
                    raise ValueError("Manifest records must be JSON objects")
                yield item
                buffer = buffer[end:]
                whitespace()
                if buffer.startswith("]"):
                    buffer = buffer[1:]
                    break
                if not buffer.startswith(","):
                    raise ValueError("Expected comma or closing bracket in manifest")
                buffer = buffer[1:]
                whitespace()
        whitespace()
        if buffer:
            raise ValueError("Unexpected data after JSON array")
