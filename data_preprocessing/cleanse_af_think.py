import json
import argparse


def cleanse(input_path, output_path, max_samples=None):
    with open(input_path) as f:
        data = json.load(f)

    items = data['data']
    if max_samples is not None:
        items = items[:max_samples]

    count_prefix = 0
    count_trailing = 0

    for item in items:
        for msg in item.get('messages', []):
            if msg.get('role') != 'user':
                continue
            content = msg['content']
            # Remove <sound>\n prefix
            if content.startswith('<sound>\n'):
                content = content[len('<sound>\n'):]
                count_prefix += 1
            # Remove trailing \n<sound> or <sound>
            if content.endswith('\n<sound>'):
                content = content[:-len('\n<sound>')]
                count_trailing += 1
            elif content.endswith('<sound>'):
                content = content[:-len('<sound>')]
                count_trailing += 1
            # Ensure content starts with <audio>
            if not content.startswith('<audio>'):
                content = '<audio>' + content
            msg['content'] = content

    print(f"Replaced '<sound>\\n' prefix with '<audio>': {count_prefix}")
    print(f"Removed trailing '<sound>': {count_trailing}")
    print(f"Total modified: {count_prefix + count_trailing}")

    out_data = {'data': items}
    with open(output_path, 'w') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of items processed (for inspection)')
    args = parser.parse_args()
    cleanse(args.input, args.output, args.max_samples)
