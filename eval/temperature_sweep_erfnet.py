"""
temperature_sweep_eomt.py
--------------------------

Expected logit filename convention:
    {dataset_name}_{image_index}_logits.pt
    e.g. FS_LostFound_full_0_logits.pt


Usage:
    python temperature_sweep_erfnet.py \
        --logits-dir /path/to/logits \
        --dataset-dir /path/to/Validation_Dataset \
        --output results_eomt_temp_sweep.csv
"""

import os
import re
import glob
import argparse
import numpy as np
import csv
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize
from sklearn.metrics import average_precision_score, roc_curve

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE = (512, 1024)
target_transform = Compose([Resize(TARGET_SIZE, Image.NEAREST)])

DATASET_REMAPS = {
    "RoadAnomaly21":     lambda gt: np.where(gt == 2, 1, gt),
    "RoadAnomaly":       lambda gt: np.where(gt == 2, 1, gt),
    "FS_LostFound_full": lambda gt: gt,
    "fs_static":         lambda gt: gt,
    "RoadObsticle21":    lambda gt: gt,
}

# ── Scoring ───────────────────────────────────────────────────────────────────

def softmax(logits):
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=0, keepdims=True)

def msp_anomaly_score(logits):
    return 1.0 - softmax(logits).max(axis=0)

# ── Metrics ───────────────────────────────────────────────────────────────────

def fpr_at_95_tpr(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    if all(tpr < 0.95):
        return 0.0
    elif all(tpr >= 0.95):
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return float(min(fpr[idx] for idx in idxs))
    else:
        return float(np.interp(0.95, tpr, fpr))

def compute_metrics(scores, labels):
    auprc = average_precision_score(labels, scores) * 100.0
    fpr95 = fpr_at_95_tpr(scores, labels) * 100.0
    return auprc, fpr95

# ── File helpers ──────────────────────────────────────────────────────────────

def parse_logit_filename(filename):
    stem = os.path.splitext(filename)[0]
    if not stem.endswith("_logits"):
        return None, None
    stem = stem[:-len("_logits")]
    match = re.match(r"^(.+)_(\d+)$", stem)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))

def load_gt_mask(dataset_name, image_index, dataset_dir):
    mask_path = os.path.join(
        dataset_dir, dataset_name, "labels_masks", f"{image_index}.png"
    )
    if not os.path.exists(mask_path):
        return None
    mask = Image.open(mask_path)
    mask = target_transform(mask)
    gt = np.array(mask)
    remap_fn = DATASET_REMAPS.get(dataset_name)
    if remap_fn:
        gt = remap_fn(gt)
    return gt

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temperature scaling sweep for EoMT-Cityscapes logits using MSP."
    )
    parser.add_argument('--logits-dir', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument('--output', default='results_eomt_temp_sweep.csv')
    args = parser.parse_args()

    temperatures = args.temperatures
    print(f"\nLogits dir  : {args.logits_dir}")
    print(f"Dataset dir : {args.dataset_dir}")
    print(f"Temperatures: {temperatures}\n")

    pt_files = sorted(glob.glob(os.path.join(args.logits_dir, "*_logits.pt")))
    if not pt_files:
        print(f"ERROR: No *_logits.pt files found in {args.logits_dir}")
        return
    print(f"Found {len(pt_files)} .pt files.\n")

    # accumulators: {dataset: {T: {'scores': [], 'labels': []}}}
    # plus 'overall' key
    acc = {}

    total = len(pt_files)
    skipped = 0

    for i, pt_path in enumerate(pt_files):
        filename = os.path.basename(pt_path)
        dataset_name, image_index = parse_logit_filename(filename)

        if dataset_name is None:
            print(f"  [{i+1}/{total}] WARNING: could not parse {filename}, skipping.")
            skipped += 1
            continue

        gt = load_gt_mask(dataset_name, image_index, args.dataset_dir)
        if gt is None:
            print(f"  [{i+1}/{total}] WARNING: GT mask not found for {filename}, skipping.")
            skipped += 1
            continue

        if 1 not in np.unique(gt):
            skipped += 1
            continue

        # ── Load .pt file ONCE ────────────────────────────────────────────────
        print(f"  [{i+1}/{total}] Loading {filename}...")
        tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
        logits = tensor.numpy().astype(np.float32)
        del tensor  # free memory immediately

        valid_mask  = (gt != 255)
        labels_flat = gt[valid_mask].ravel().astype(np.int32)

        # ── Sweep all temperatures on this file ───────────────────────────────
        for T in temperatures:
            scaled   = logits / T                          # [19, H, W]
            scores   = msp_anomaly_score(scaled)           # [H, W]
            scores_flat = scores[valid_mask].ravel()

            # initialise accumulators if needed
            if dataset_name not in acc:
                acc[dataset_name] = {t: {'scores': [], 'labels': []} for t in temperatures}
            if 'overall' not in acc:
                acc['overall'] = {t: {'scores': [], 'labels': []} for t in temperatures}

            acc[dataset_name][T]['scores'].append(scores_flat)
            acc[dataset_name][T]['labels'].append(labels_flat)
            acc['overall'][T]['scores'].append(scores_flat)
            acc['overall'][T]['labels'].append(labels_flat)

        del logits  # free memory before next file
        # ─────────────────────────────────────────────────────────────────────

    print(f"\nDone loading. {skipped} files skipped.\n")
    print("Computing metrics...\n")

    datasets = sorted(k for k in acc.keys() if k != 'overall')

    # ── Print results table ───────────────────────────────────────────────────
    col = 10
    header = f"{'T':>6}  "
    for ds in datasets:
        short = ds[:8]
        header += f"{short+' AuPRC':>{col}}  {short+' FPR95':>{col}}  "
    header += f"{'All AuPRC':>{col}}  {'All FPR95':>{col}}"
    print(header)
    print("-" * len(header))

    csv_rows = []
    best = {ds: {'auprc': -1, 'fpr95': 999, 'T_auprc': None, 'T_fpr95': None}
            for ds in datasets + ['overall']}

    for T in temperatures:
        row  = {'temperature': T}
        line = f"{T:>6.3f}  "

        for ds in datasets:
            s = np.concatenate(acc[ds][T]['scores'])
            l = np.concatenate(acc[ds][T]['labels'])
            auprc, fpr95 = compute_metrics(s, l)
            row[f'{ds}_AUPRC'] = round(auprc, 2)
            row[f'{ds}_FPR95'] = round(fpr95, 2)
            line += f"{auprc:>{col}.2f}  {fpr95:>{col}.2f}  "

            if auprc > best[ds]['auprc']:
                best[ds]['auprc'] = auprc; best[ds]['T_auprc'] = T
            if fpr95 < best[ds]['fpr95']:
                best[ds]['fpr95'] = fpr95; best[ds]['T_fpr95'] = T

        s = np.concatenate(acc['overall'][T]['scores'])
        l = np.concatenate(acc['overall'][T]['labels'])
        oa, of = compute_metrics(s, l)
        row['overall_AUPRC'] = round(oa, 2)
        row['overall_FPR95'] = round(of, 2)
        line += f"{oa:>{col}.2f}  {of:>{col}.2f}"
        print(line)
        csv_rows.append(row)

        if oa > best['overall']['auprc']:
            best['overall']['auprc'] = oa; best['overall']['T_auprc'] = T
        if of < best['overall']['fpr95']:
            best['overall']['fpr95'] = of; best['overall']['T_fpr95'] = T

    # ── Best summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BEST RESULTS:")
    for ds in datasets:
        print(f"\n  {ds}:")
        print(f"    Best AUPRC : {best[ds]['auprc']:.2f}%  at T={best[ds]['T_auprc']}")
        print(f"    Best FPR95 : {best[ds]['fpr95']:.2f}%  at T={best[ds]['T_fpr95']}")
    print(f"\n  Overall:")
    print(f"    Best AUPRC : {best['overall']['auprc']:.2f}%  at T={best['overall']['T_auprc']}")
    print(f"    Best FPR95 : {best['overall']['fpr95']:.2f}%  at T={best['overall']['T_fpr95']}")
    print(f"\n  (T=1.0 = baseline, no scaling)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    fieldnames = ['temperature']
    for ds in datasets:
        fieldnames += [f'{ds}_AUPRC', f'{ds}_FPR95']
    fieldnames += ['overall_AUPRC', 'overall_FPR95']

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults saved to {args.output}\n")


if __name__ == '__main__':
    main()