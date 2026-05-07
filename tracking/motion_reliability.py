import json
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


class MotionReliabilityCalibrator:
    """
    Lightweight confidence-aware motion reliability calibrator.

    Supported modes:
    - manual: analytical sigmoid baseline
    - learned_linear: logistic regression over normalized continuous features
    - tiny_mlp: shallow 1-hidden-layer perceptron loaded from a json checkpoint
    """

    DEFAULT_FEATURES = [
        'det_score',
        'track_uncertainty',
        'track_tsu',
        'track_beta',
        'track_hits',
        'track_age',
        'track_speed',
        'pair_center_dist',
        'vel_sim',
    ]

    def __init__(
        self,
        mode='manual',
        det_thresh=0.2,
        manual_bias=-0.2,
        manual_score_gain=2.5,
        manual_uncertainty_gain=2.0,
        feature_names: Optional[Iterable[str]] = None,
        weight_path: str = '',
        linear_weights: Optional[Iterable[float]] = None,
        linear_bias: float = 0.0,
        mlp_hidden_weights: Optional[Iterable[Iterable[float]]] = None,
        mlp_hidden_bias: Optional[Iterable[float]] = None,
        mlp_output_weights: Optional[Iterable[float]] = None,
        mlp_output_bias: float = 0.0,
    ):
        self.mode = str(mode or 'manual').strip().lower()
        self.det_thresh = float(det_thresh)
        self.manual_bias = float(manual_bias)
        self.manual_score_gain = float(manual_score_gain)
        self.manual_uncertainty_gain = float(manual_uncertainty_gain)
        self.feature_names = list(feature_names or self.DEFAULT_FEATURES)
        self.weight_path = str(weight_path or '').strip()
        self.linear_weights = None if linear_weights is None else np.asarray(linear_weights, dtype=np.float32)
        self.linear_bias = float(linear_bias)
        self.mlp_hidden_weights = None if mlp_hidden_weights is None else np.asarray(mlp_hidden_weights, dtype=np.float32)
        self.mlp_hidden_bias = None if mlp_hidden_bias is None else np.asarray(mlp_hidden_bias, dtype=np.float32)
        self.mlp_output_weights = None if mlp_output_weights is None else np.asarray(mlp_output_weights, dtype=np.float32)
        self.mlp_output_bias = float(mlp_output_bias)
        if self.weight_path:
            self._load_weights(self.weight_path)

    def _load_weights(self, weight_path):
        if not weight_path or not os.path.isfile(weight_path):
            return
        with open(weight_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.mode = str(payload.get('mode', self.mode)).strip().lower()
        self.feature_names = list(payload.get('feature_names', self.feature_names))
        if 'det_thresh' in payload:
            self.det_thresh = float(payload['det_thresh'])
        if 'manual_bias' in payload:
            self.manual_bias = float(payload['manual_bias'])
        if 'manual_score_gain' in payload:
            self.manual_score_gain = float(payload['manual_score_gain'])
        if 'manual_uncertainty_gain' in payload:
            self.manual_uncertainty_gain = float(payload['manual_uncertainty_gain'])
        if 'linear_weights' in payload:
            self.linear_weights = np.asarray(payload['linear_weights'], dtype=np.float32)
        if 'linear_bias' in payload:
            self.linear_bias = float(payload['linear_bias'])
        if 'mlp_hidden_weights' in payload:
            self.mlp_hidden_weights = np.asarray(payload['mlp_hidden_weights'], dtype=np.float32)
        if 'mlp_hidden_bias' in payload:
            self.mlp_hidden_bias = np.asarray(payload['mlp_hidden_bias'], dtype=np.float32)
        if 'mlp_output_weights' in payload:
            self.mlp_output_weights = np.asarray(payload['mlp_output_weights'], dtype=np.float32)
        if 'mlp_output_bias' in payload:
            self.mlp_output_bias = float(payload['mlp_output_bias'])

    def signature(self):
        return (
            self.mode,
            self.det_thresh,
            self.manual_bias,
            self.manual_score_gain,
            self.manual_uncertainty_gain,
            tuple(self.feature_names),
            self.weight_path,
        )

    def vectorize(self, feature_dict: Dict[str, float]):
        values: List[float] = []
        for name in self.feature_names:
            values.append(float(feature_dict.get(name, 0.0)))
        return np.asarray(values, dtype=np.float32)

    def predict(self, feature_dict: Dict[str, float]):
        det_score = _clip01(feature_dict.get('det_score', 1.0))
        track_uncertainty = _clip01(feature_dict.get('track_uncertainty', 0.0))
        if self.mode == 'manual':
            logit = (
                self.manual_bias +
                self.manual_score_gain * (det_score - self.det_thresh) -
                self.manual_uncertainty_gain * track_uncertainty
            )
            return _clip01(_sigmoid(logit))

        x = self.vectorize(feature_dict)
        if self.mode == 'learned_linear':
            if self.linear_weights is None or self.linear_weights.shape[0] != x.shape[0]:
                return self.predict_manual_fallback(det_score, track_uncertainty)
            return _clip01(_sigmoid(np.dot(self.linear_weights, x) + self.linear_bias))

        if self.mode == 'tiny_mlp':
            if (
                self.mlp_hidden_weights is None or
                self.mlp_hidden_bias is None or
                self.mlp_output_weights is None or
                self.mlp_hidden_weights.shape[1] != x.shape[0]
            ):
                return self.predict_manual_fallback(det_score, track_uncertainty)
            h = np.maximum(0.0, np.dot(self.mlp_hidden_weights, x) + self.mlp_hidden_bias)
            return _clip01(_sigmoid(np.dot(self.mlp_output_weights, h) + self.mlp_output_bias))

        return self.predict_manual_fallback(det_score, track_uncertainty)

    def predict_manual_fallback(self, det_score, track_uncertainty):
        logit = (
            self.manual_bias +
            self.manual_score_gain * (float(det_score) - self.det_thresh) -
            self.manual_uncertainty_gain * float(track_uncertainty)
        )
        return _clip01(_sigmoid(logit))

    @staticmethod
    def build_feature_dict(
        det_score=1.0,
        track_uncertainty=0.0,
        track_tsu=0.0,
        track_beta=1.0,
        track_hits=0.0,
        track_age=0.0,
        track_speed=0.0,
        pair_center_dist=0.0,
        vel_sim=0.5,
    ):
        return {
            'det_score': _clip01(det_score),
            'track_uncertainty': _clip01(track_uncertainty),
            'track_tsu': _clip01(track_tsu),
            'track_beta': _clip01(track_beta),
            'track_hits': _clip01(track_hits),
            'track_age': _clip01(track_age),
            'track_speed': _clip01(track_speed),
            'pair_center_dist': _clip01(pair_center_dist),
            'vel_sim': _clip01(vel_sim),
        }


def compute_box_iou_xyxy(box_a, box_b):
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except Exception:
        return 0.0
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def load_kitti_tracking_gt_by_frame(label_path, allowed_classes=None):
    gt_by_frame = {}
    if not label_path or not os.path.isfile(label_path):
        return gt_by_frame
    allowed = None
    if allowed_classes:
        allowed = {str(name).strip().lower() for name in allowed_classes}
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            try:
                frame = int(parts[0])
                track_id = int(parts[1])
                class_name = str(parts[2])
                bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
            except Exception:
                continue
            if track_id < 0:
                continue
            if allowed is not None and class_name.strip().lower() not in allowed:
                continue
            gt_by_frame.setdefault(frame, []).append({
                'track_id': track_id,
                'class_name': class_name,
                'bbox_2d': bbox,
            })
    return gt_by_frame


def assign_detection_gt_ids(detections, gt_entries, min_iou=0.5):
    if detections is None:
        return
    for det in detections:
        try:
            det.gt_track_id = -1
            det.gt_class_name = ''
            det.gt_iou = 0.0
        except Exception:
            pass
    if not detections or not gt_entries:
        return
    matches = []
    for di, det in enumerate(detections):
        try:
            det_box = np.asarray(det.additional_info[2:6], dtype=np.float32).tolist()
        except Exception:
            continue
        for gi, gt in enumerate(gt_entries):
            iou = compute_box_iou_xyxy(det_box, gt.get('bbox_2d', None))
            if iou >= float(min_iou):
                matches.append((float(iou), di, gi))
    matches.sort(reverse=True)
    used_dets = set()
    used_gts = set()
    for iou, di, gi in matches:
        if di in used_dets or gi in used_gts:
            continue
        used_dets.add(di)
        used_gts.add(gi)
        try:
            detections[di].gt_track_id = int(gt_entries[gi].get('track_id', -1))
            detections[di].gt_class_name = str(gt_entries[gi].get('class_name', ''))
            detections[di].gt_iou = float(iou)
        except Exception:
            continue


class MotionReliabilitySampleExporter:
    def __init__(self, output_path='', enabled=False):
        self.output_path = str(output_path or '').strip()
        self.enabled = bool(enabled) and bool(self.output_path)

    def export(self, sample):
        if not self.enabled:
            return
        try:
            out_dir = os.path.dirname(self.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=True) + '\n')
        except Exception:
            return
