import os
import re
import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

CKPT_PATTERN = re.compile(r'^c(\d+)-ep(\d+)\.pt$')

# Test-set decision thresholds for each split. These affect only ACC, precision, recall, and MCC;
# AUROC and AUPR are threshold independent. Register new splits here by test_name.
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = {
    "Random_test": 0.50,
    "Cold_test_drug": 0.50,
    "Cold_test_target": 0.50,
    "Temporal": 0.50,
}


def run_inference(model, loader, device):
    """Run inference on a loader and return (y_true, y_score) without consuming global RNG state."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for data_sample in loader:
            y = data_sample['label'].to(device)
            outputs = model(
                data_sample['target_embed'].to(device),
                data_sample['target_tokens'].to(device),
                data_sample['warhead_graph'].to(device),
                data_sample['linker_graph'].to(device),
                data_sample['e3_ligand_graph'].to(device),
                data_sample['ligase_embed'].to(device),
                data_sample['ligase_tokens'].to(device),
                mol_descriptors=data_sample['mol_descriptors'].to(device),
            )
            probs = torch.nn.functional.softmax(outputs, dim=1)
            y_score.extend(probs[:, 1].cpu().tolist())
            y_true.extend(y.cpu().tolist())
    return y_true, y_score


def compute_metrics(y_true, y_score, threshold):
    """Compute AUROC/AUPR from scores and threshold-dependent metrics from TP/FP/TN/FN counts."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    preds = (y_score >= threshold).astype(np.int32)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    return {
        "acc": float(acc), "pre": float(pre), "recall": float(recall),
        "auroc": float(auroc), "aupr": float(aupr), "mcc": float(mcc),
    }


def evaluate_checkpoints(model, ckpt_dir, test_loader, test_name, device):
    """
    Load each c*.pt checkpoint from ckpt_dir and evaluate it on the independent test set.
    Report the checkpoint with the highest test AUROC as the best model.
    Resolve the decision threshold from THRESHOLDS by test_name, or use DEFAULT_THRESHOLD.
    """
    threshold = THRESHOLDS.get(test_name, DEFAULT_THRESHOLD)
    files = [f for f in os.listdir(ckpt_dir) if CKPT_PATTERN.match(f)]
    files.sort(key=lambda f: int(CKPT_PATTERN.match(f).group(1)))  # Sort by checkpoint index.

    best_auroc = float('-inf')
    best_file = None
    best_metrics = None

    for fname in files:
        state_dict = torch.load(os.path.join(ckpt_dir, fname), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        y_true, y_score = run_inference(model, test_loader, device)
        m = compute_metrics(y_true, y_score, threshold)
        print(
            f"  [eval] {fname}: ACC={m['acc']:.6f}  |  Pre={m['pre']:.6f}  |  "
            f"Recall={m['recall']:.6f}  |  AUROC={m['auroc']:.6f}  |  "
            f"AUPR={m['aupr']:.6f}  |  MCC={m['mcc']:.6f}"
        )
        if m['auroc'] > best_auroc:
            best_auroc = m['auroc']
            best_file = fname
            best_metrics = m

    if best_metrics is not None:
        logging.info("✅ TRAINING COMPLETE")
        logging.info(f"best-model: {best_file}")
        logging.info(
            f"{test_name}：ACC：{best_metrics['acc']:.6f}  |  "
            f"Precision：{best_metrics['pre']:.6f}  |  "
            f"Recall：{best_metrics['recall']:.6f}  |  "
            f"AUROC：{best_metrics['auroc']:.6f}  |  "
            f"AUPR：{best_metrics['aupr']:.6f}  |  "
            f"MCC：{best_metrics['mcc']:.6f}"
        )

    return best_file, best_metrics
