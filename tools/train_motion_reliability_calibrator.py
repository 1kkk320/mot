import argparse
import json
import os

import numpy as np


def load_samples(path):
    features = []
    labels = []
    meta = {'total': 0, 'used': 0}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta['total'] += 1
            item = json.loads(line)
            if 'feature_dict' not in item or 'label' not in item:
                continue
            features.append(item['feature_dict'])
            labels.append(int(item['label']))
            meta['used'] += 1
    return features, np.asarray(labels, dtype=np.float32), meta


def vectorize(feature_dicts, feature_names):
    x = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float32)
    for i, feature_dict in enumerate(feature_dicts):
        for j, name in enumerate(feature_names):
            x[i, j] = float(feature_dict.get(name, 0.0))
    return x


def train_logreg(x, y, lr=0.1, epochs=400, l2=1e-4):
    n, d = x.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0
    pos = max(float(np.sum(y > 0.5)), 1.0)
    neg = max(float(np.sum(y <= 0.5)), 1.0)
    pos_weight = neg / pos
    for _ in range(int(epochs)):
        logits = x @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        weights = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
        error = (probs - y) * weights
        grad_w = (x.T @ error) / max(float(n), 1.0) + float(l2) * w
        grad_b = float(np.mean(error))
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w, b


def evaluate(x, y, w, b):
    logits = x @ w + b
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    preds = (probs >= 0.5).astype(np.float32)
    eps = 1e-6
    loss = float(-np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))) if y.size else 0.0
    acc = float(np.mean(preds == y)) if y.size else 0.0
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    recall = float(np.mean(preds[pos_mask] == 1.0)) if np.any(pos_mask) else 0.0
    specificity = float(np.mean(preds[neg_mask] == 0.0)) if np.any(neg_mask) else 0.0
    return {
        'loss': loss,
        'accuracy': acc,
        'recall_pos': recall,
        'specificity_neg': specificity,
        'num_samples': int(y.size),
        'num_pos': int(np.sum(pos_mask)),
        'num_neg': int(np.sum(neg_mask)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train a lightweight learned-linear motion reliability calibrator.')
    parser.add_argument('--samples', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', default='det_score,track_uncertainty,track_tsu,track_beta,track_hits,track_age,track_speed,pair_center_dist,vel_sim')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=1e-4)
    args = parser.parse_args()

    feature_names = [item.strip() for item in str(args.features).split(',') if item.strip()]
    feature_dicts, y, meta = load_samples(args.samples)
    x = vectorize(feature_dicts, feature_names)
    w, b = train_logreg(x, y, lr=args.lr, epochs=args.epochs, l2=args.l2)
    metrics = evaluate(x, y, w, b)

    payload = {
        'mode': 'learned_linear',
        'feature_names': feature_names,
        'linear_weights': [float(v) for v in w.tolist()],
        'linear_bias': float(b),
        'train_metrics': metrics,
        'sample_meta': meta,
    }
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print('Saved calibrator to:', args.output)
    print('Sample meta:', meta)
    print('Metrics:', metrics)


if __name__ == '__main__':
    main()
