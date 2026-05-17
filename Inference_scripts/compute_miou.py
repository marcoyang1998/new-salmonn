import json
import re
import numpy as np
import torch
from nncore.ops import temporal_area, temporal_intersection
import argparse
import os


def compute_iou_multi(pred, span):
    try:
        pred_tensor = torch.Tensor(pred)
        span_tensor = torch.Tensor(span)
        pred_area = temporal_area(pred_tensor).sum()
        span_area = temporal_area(span_tensor).sum()
        inter = temporal_intersection(pred_tensor, span_tensor).sum()
        iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
        iou = torch.where(iou.isfinite(), iou, 0)
        return iou
    except:
        return torch.tensor([0.])


def parse_intervals(text):
    """Parse [[a1, b1], [a2, b2], ...] from text, handling <|X.XX|> tokens."""
    if not text or text.strip() == '[]':
        return []
    # Extract all numbers (handles both <|0.00|> format and plain floats)
    numbers = re.findall(r'(\d+\.?\d*)', text)
    intervals = []
    for i in range(0, len(numbers) - 1, 2):
        start = float(numbers[i])
        end = float(numbers[i + 1])
        if start < end:
            intervals.append([start, end])
    return intervals


def main():
    parser = argparse.ArgumentParser(description='Compute mIOU for temporal predictions')
    parser.add_argument('--prediction_file', '-p', type=str, help='Path to prediction JSON file')
    args = parser.parse_args()

    with open(args.prediction_file, 'r') as f:
        data = json.load(f)

    items = data['data']
    ious = []
    empty_pred_count = 0
    empty_gt_count = 0

    for item in items:
        gt_text = item['messages'][1]['content']
        pred_text = item['model_prediction']

        # pred_text = pred_text[len("user\nWhen does the following event happen: speech? Answer in the format of [[a1, b1], [a2, b2], ...]\nassistant\n"):]

        gt_intervals = parse_intervals(gt_text)
        pred_intervals = parse_intervals(pred_text)

        if len(gt_intervals) == 0:
            empty_gt_count += 1
            continue
        if len(pred_intervals) == 0:
            empty_pred_count += 1
            ious.append(0.0)
            continue

        iou = compute_iou_multi(pred_intervals, gt_intervals).item()
        ious.append(iou)

    ious = np.array(ious)
    print(f'Total samples: {len(items)}')
    print(f'Valid samples (non-empty GT): {len(ious)}')
    print(f'Empty GT skipped: {empty_gt_count}')
    print(f'Empty prediction (IoU=0): {empty_pred_count}')
    print(f'mIOU: {ious.mean():.4f}')
    print(f'Median IoU: {np.median(ious):.4f}')
    print(f'IoU > 0.5: {(ious > 0.5).sum()} / {len(ious)} ({(ious > 0.5).mean() * 100:.1f}%)')
    print(f'IoU > 0.3: {(ious > 0.3).sum()} / {len(ious)} ({(ious > 0.3).mean() * 100:.1f}%)')


if __name__ == '__main__':
    main()
