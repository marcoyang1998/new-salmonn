import argparse
import json
import math
import re
from collections import defaultdict


DIMENSIONS = [
    "Noise",
    "Distortion",
    "Speed",
    "Continuity",
    "Effort",
    "Naturalness",
    "Overall",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute QualiSpeech PCC by quality dimension")
    parser.add_argument(
        "--result_json",
        type=str,
        required=True,
        help="Path to QualiSpeech result JSON with model_prediction fields",
    )
    return parser.parse_args()


def get_items(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected result JSON to be a list or a dict containing a 'data' list.")


def get_message_content(item, role):
    for message in item.get("messages", []):
        if message.get("role") == role:
            return str(message.get("content", "")).strip()
    return ""


def infer_dimension(prompt):
    text = prompt.lower()
    if "noise" in text:
        return "Noise"
    if "distortion" in text:
        return "Distortion"
    if "speed" in text:
        return "Speed"
    if "continuity" in text or "smoothness" in text:
        return "Continuity"
    if "effort" in text:
        return "Effort"
    if "naturalness" in text or "natural" in text:
        return "Naturalness"
    if "overall" in text:
        return "Overall"
    return None


def parse_score(value):
    match = re.fullmatch(r"\s*([1-5](?:\.0+)?)\s*", str(value))
    if match is None:
        return None
    return float(match.group(1))


def pearson_correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return math.nan

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denom_x = sum(x * x for x in centered_x)
    denom_y = sum(y * y for y in centered_y)
    denominator = math.sqrt(denom_x * denom_y)
    if denominator == 0:
        return math.nan
    return numerator / denominator


def main():
    args = parse_args()
    with open(args.result_json, "r") as f:
        payload = json.load(f)

    grouped = defaultdict(lambda: {"refs": [], "preds": [], "skipped": 0})
    unknown_dimension = 0

    for item in get_items(payload):
        prompt = get_message_content(item, "user")
        dimension = infer_dimension(prompt)
        if dimension is None:
            unknown_dimension += 1
            continue

        reference = parse_score(get_message_content(item, "assistant"))
        prediction = parse_score(item.get("model_prediction", ""))
        if reference is None or prediction is None:
            grouped[dimension]["skipped"] += 1
            continue

        grouped[dimension]["refs"].append(reference)
        grouped[dimension]["preds"].append(prediction)

    print("QualiSpeech PCC by dimension")
    print(f"Result JSON: {args.result_json}")
    print("")
    print(f"{'Dimension':<13} {'PCC':>10} {'N':>8} {'Skipped':>8}")
    print("-" * 43)
    for dimension in DIMENSIONS:
        refs = grouped[dimension]["refs"]
        preds = grouped[dimension]["preds"]
        pcc = pearson_correlation(refs, preds)
        pcc_text = "nan" if math.isnan(pcc) else f"{pcc:.6f}"
        print(f"{dimension:<13} {pcc_text:>10} {len(refs):>8} {grouped[dimension]['skipped']:>8}")

    if unknown_dimension:
        print("")
        print(f"Unknown dimension items skipped: {unknown_dimension}")


if __name__ == "__main__":
    main()
