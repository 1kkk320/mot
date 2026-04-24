# -*-coding:utf-8-*
# author: wangxy
import numpy as np
import math
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance
import scipy.spatial as sp

def compute_rotated_ground_similarity(det_box, trk_box, det_corners=None, trk_corners=None):
    """
    Continuous ground-plane geometric similarity using the existing 3D boxes.
    This is a lightweight MCTrack-style replacement for plain 3D IoU scoring:
    it jointly reflects rotated BEV overlap, center alignment and size consistency.
    """
    try:
        if det_corners is None:
            det_corners = convert_3dbox_to_8corner(det_box)
        if trk_corners is None:
            trk_corners = convert_3dbox_to_8corner(trk_box)
        _, bev_iou = iou3d(det_corners, trk_corners)
        bev_iou = float(np.clip(bev_iou, 0.0, 1.0))

        det_box = np.asarray(det_box, dtype=np.float32)
        trk_box = np.asarray(trk_box, dtype=np.float32)

        center_dist = float(np.linalg.norm(det_box[[0, 2]] - trk_box[[0, 2]]))
        det_diag = float(np.hypot(max(det_box[4], 1e-3), max(det_box[5], 1e-3)))
        trk_diag = float(np.hypot(max(trk_box[4], 1e-3), max(trk_box[5], 1e-3)))
        scale_ref = max(0.5 * (det_diag + trk_diag), 1e-3)
        center_sim = float(np.exp(-center_dist / scale_ref))

        size_delta = 0.5 * (
            abs(math.log(max(float(det_box[4]), 1e-3) / max(float(trk_box[4]), 1e-3))) +
            abs(math.log(max(float(det_box[5]), 1e-3) / max(float(trk_box[5]), 1e-3)))
        )
        size_sim = float(np.exp(-size_delta))

        return float((bev_iou + center_sim + size_sim) / 3.0)
    except Exception:
        return 0.0


def _safe_sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_track_uncertainty(track, uncertainty_norm=12.0):
    try:
        P = track.kf_3d.kf.P
        sx = float(np.sqrt(np.abs(P[0, 0]))) if P.shape[0] > 0 else 0.0
        sz = float(np.sqrt(np.abs(P[2, 2]))) if P.shape[0] > 2 else 0.0
        return float(np.clip((sx + sz) / max(float(uncertainty_norm), 1e-6), 0.0, 1.0))
    except Exception:
        return 0.0


def _top2_margin(valid_scores):
    if valid_scores.size <= 1:
        return np.nan
    sorted_scores = np.sort(valid_scores)[::-1]
    return float(sorted_scores[0] - sorted_scores[1])


def compute_association_risk_context(
    score_matrix,
    candidate_mask,
    detections,
    trackers,
    emb_cost=None,
    risk_options=None,
):
    valid_mask = candidate_mask.astype(bool)
    pair_risk = np.zeros_like(score_matrix, dtype=np.float32)
    identity_support = np.full_like(score_matrix, 0.5, dtype=np.float32)
    row_ambiguity = np.zeros((score_matrix.shape[0],), dtype=np.float32)
    col_ambiguity = np.zeros((score_matrix.shape[1],), dtype=np.float32)
    is_active = False
    if risk_options is not None:
        is_active = bool(risk_options.get('enabled', False)) or bool(risk_options.get('enable_tempering', False)) or bool(risk_options.get('enable_deferral', False))
    if not is_active or not np.any(valid_mask):
        return pair_risk, identity_support, row_ambiguity, col_ambiguity

    risk_center = float(risk_options.get('margin_center', 0.08))
    risk_gain = float(risk_options.get('margin_gain', 18.0))
    score_center = float(risk_options.get('score_center', 0.45))
    score_gain = float(risk_options.get('score_gain', 8.0))
    tsu_center = float(risk_options.get('tsu_center', 1.5))
    tsu_gain = float(risk_options.get('tsu_gain', 3.0))
    beta_center = float(risk_options.get('beta_center', 0.55))
    beta_gain = float(risk_options.get('beta_gain', 8.0))
    uncertainty_center = float(risk_options.get('uncertainty_center', 0.25))
    uncertainty_gain = float(risk_options.get('uncertainty_gain', 8.0))
    uncertainty_norm = float(risk_options.get('uncertainty_norm', 12.0))
    temper_strength = float(risk_options.get('temper_strength', 0.20))
    app_support_center = float(risk_options.get('app_support_center', 0.55))
    app_support_gain = float(risk_options.get('app_support_gain', 10.0))
    app_rescue = float(risk_options.get('app_rescue', 0.06))

    for d in range(score_matrix.shape[0]):
        valid_scores = score_matrix[d, valid_mask[d]]
        margin = _top2_margin(valid_scores)
        if np.isnan(margin):
            row_ambiguity[d] = 0.0
        else:
            row_ambiguity[d] = float(_safe_sigmoid(risk_gain * (risk_center - margin)))

    for t in range(score_matrix.shape[1]):
        valid_scores = score_matrix[valid_mask[:, t], t]
        margin = _top2_margin(valid_scores)
        if np.isnan(margin):
            col_ambiguity[t] = 0.0
        else:
            col_ambiguity[t] = float(_safe_sigmoid(risk_gain * (risk_center - margin)))

    det_risk = np.zeros((score_matrix.shape[0],), dtype=np.float32)
    for d, det in enumerate(detections):
        try:
            det_score = float(getattr(det, 'score', 1.0))
        except Exception:
            det_score = 1.0
        det_score = float(np.clip(det_score, 0.0, 1.0))
        det_risk[d] = float(_safe_sigmoid(score_gain * (score_center - det_score)))

    trk_risk = np.zeros((score_matrix.shape[1],), dtype=np.float32)
    for t, trk in enumerate(trackers):
        tsu = float(max(0, int(getattr(trk, 'time_since_update', 0))))
        beta = float(np.clip(getattr(trk, 'beta_t', 1.0), 0.0, 1.0))
        uncertainty = _compute_track_uncertainty(trk, uncertainty_norm=uncertainty_norm)
        tsu_risk = float(_safe_sigmoid(tsu_gain * (tsu - tsu_center)))
        beta_risk = float(_safe_sigmoid(beta_gain * (beta_center - beta)))
        uncertainty_risk = float(_safe_sigmoid(uncertainty_gain * (uncertainty - uncertainty_center)))
        trk_risk[t] = float(np.clip(
            0.25 * tsu_risk + 0.35 * beta_risk + 0.40 * uncertainty_risk,
            0.0,
            1.0,
        ))

    for d in range(score_matrix.shape[0]):
        for t in range(score_matrix.shape[1]):
            if not valid_mask[d, t]:
                continue
            ambiguity = max(float(row_ambiguity[d]), float(col_ambiguity[t]))
            context_risk = 0.55 * float(trk_risk[t]) + 0.45 * float(det_risk[d])
            pair_risk[d, t] = float(np.clip(ambiguity * context_risk, 0.0, 1.0))

    if emb_cost is not None and emb_cost.shape == score_matrix.shape:
        identity_support = _safe_sigmoid(app_support_gain * (emb_cost - app_support_center)).astype(np.float32)
    return pair_risk, identity_support, row_ambiguity, col_ambiguity


def apply_association_risk_tempering(
    score_matrix,
    pair_risk,
    identity_support,
    risk_options=None,
):
    if risk_options is None or not risk_options.get('enable_tempering', False):
        return score_matrix

    temper_strength = float(risk_options.get('temper_strength', 0.20))
    app_rescue = float(risk_options.get('app_rescue', 0.06))
    score_matrix = score_matrix.copy()
    attenuation = temper_strength * pair_risk * (1.0 - identity_support)
    score_matrix = score_matrix * (1.0 - attenuation)
    score_matrix = score_matrix + app_rescue * pair_risk * identity_support
    return score_matrix


def apply_deferred_commitment(
    matched_indices,
    score_matrix,
    candidate_mask,
    pair_risk,
    identity_support,
    row_ambiguity,
    col_ambiguity,
    options=None,
):
    diagnostics = {
        'initial_match_count': int(matched_indices.shape[0]) if matched_indices.ndim == 2 else 0,
        'final_match_count': int(matched_indices.shape[0]) if matched_indices.ndim == 2 else 0,
        'deferred_pairs': [],
        'deferred_det_indices': [],
        'deferred_trk_indices': [],
    }
    if options is None or not options.get('enable_deferral', False) or matched_indices.size == 0:
        return matched_indices, diagnostics

    score_center = float(options.get('defer_score_center', 0.25))
    score_gain = float(options.get('defer_score_gain', 10.0))
    ambiguity_center = float(options.get('defer_ambiguity_center', 0.55))
    ambiguity_gain = float(options.get('defer_ambiguity_gain', 10.0))
    defer_threshold = float(options.get('defer_threshold', 0.16))
    identity_floor = float(options.get('identity_floor', 0.60))

    kept_matches = []
    for m in matched_indices:
        d, t = int(m[0]), int(m[1])
        if candidate_mask[d, t] == 0:
            continue
        ambiguity = max(float(row_ambiguity[d]), float(col_ambiguity[t]))
        score_focus = float(_safe_sigmoid(score_gain * (float(score_matrix[d, t]) - score_center)))
        ambiguity_focus = float(_safe_sigmoid(ambiguity_gain * (ambiguity - ambiguity_center)))
        defer_strength = float(pair_risk[d, t]) * ambiguity_focus * score_focus * (1.0 - float(identity_support[d, t]))
        if defer_strength >= defer_threshold and float(identity_support[d, t]) < identity_floor:
            diagnostics['deferred_pairs'].append({
                'det_idx': d,
                'trk_idx': t,
                'score': float(score_matrix[d, t]),
                'pair_risk': float(pair_risk[d, t]),
                'ambiguity': float(ambiguity),
                'identity_support': float(identity_support[d, t]),
                'defer_strength': float(defer_strength),
            })
            diagnostics['deferred_det_indices'].append(d)
            diagnostics['deferred_trk_indices'].append(t)
            continue
        kept_matches.append(m.reshape(1, 2))

    if len(kept_matches) == 0:
        diagnostics['final_match_count'] = 0
        return np.empty((0, 2), dtype=int), diagnostics
    kept_matches = np.concatenate(kept_matches, axis=0)
    diagnostics['final_match_count'] = int(kept_matches.shape[0])
    return kept_matches, diagnostics
def split_cosine_dist(dets, trks, affinity_thresh=0.50, pair_diff_thresh=0.6, hard_thresh=True):

    cos_dist = np.zeros((len(dets), len(trks)))

    for i in range(len(dets)):
        for j in range(len(trks)):
            # 兼容一维向量/二维patch：统一转换为二维行向量
            det_ij = np.asarray(dets[i])
            trk_ij = np.asarray(trks[j])
            if det_ij.ndim == 1:
                det_ij = det_ij[None, :]
            if trk_ij.ndim == 1:
                trk_ij = trk_ij[None, :]
            # 统一特征维度：使用双方共同的最小维度
            d_dim = det_ij.shape[1]
            t_dim = trk_ij.shape[1]
            if d_dim != t_dim:
                min_dim = min(d_dim, t_dim)
                det_ij = det_ij[:, :min_dim]
                trk_ij = trk_ij[:, :min_dim]

            cos_d = 1 - sp.distance.cdist(det_ij, trk_ij, "cosine")  ## shape = [m_d, m_t]
            patch_affinity = np.max(cos_d, axis=0)  ## shape = [3,]
            # exp16 - Using Hard threshold
            if hard_thresh:
                if len(np.where(patch_affinity > affinity_thresh)[0]) != len(patch_affinity):
                    cos_dist[i, j] = 0
                else:
                    cos_dist[i, j] = np.max(patch_affinity)
            else:
                cos_dist[i, j] = np.max(patch_affinity)  # can experiment with mean too (max works slightly better)

    return cos_dist


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_tracks(tracks, detections, threshold, aw_off, grid_off,mot_off, det_embs=None, det_app = False):
    track_indices = list(range(len(tracks)))
    detection_indices = list(range(len(detections)))
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    if not det_app:
        trks_list = []
        for trk in tracks:
            e = np.asarray(trk.emb) if hasattr(trk, 'emb') else None
            if e is None:
                trks_list.append(None)
                continue
            if e.ndim == 2:
                e = e.mean(axis=0)
            trks_list.append(e)

    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            iou_matrix[t, d] = iou_batch(trk.x1y1x2y2(), det.to_x1y1x2y2())  # det: 8 x 3, trk: 8 x 3
            # print("更新前", iou_matrix[t, d])
            iou_matrix[t, d] = iou_matrix[t,d] / trk.confidence
            # print("更新后", iou_matrix[t, d], type(iou_matrix[t, d]), "置信度", trk.confidence)
            # if np.isinf(iou_matrix[d, t]) or np.isnan(iou_matrix[d, t]):
            #     iou_matrix[d, t] = 0
    if not det_app:
        if grid_off:
            if det_embs is None or det_embs.size == 0 or len(trks_list) == 0:
                emb_cost = None
            else:
                d_dim = det_embs.shape[1]
                trk_embs_mat = np.zeros((len(trks_list), d_dim), dtype=np.float32)
                for i, e in enumerate(trks_list):
                    if e is None:
                        continue
                    if e.ndim == 1:
                        if e.shape[0] == d_dim:
                            trk_embs_mat[i] = e
                        else:
                            m = min(d_dim, e.shape[0])
                            trk_embs_mat[i, :m] = e[:m]
                    else:
                        v = e.reshape(-1)
                        m = min(d_dim, v.shape[0])
                        trk_embs_mat[i, :m] = v[:m]
                emb_cost = trk_embs_mat @ det_embs.T
        else:
            trks_embs = np.asarray([np.asarray(e) if e is not None else np.zeros_like(det_embs[0]) for e in trks_list])
            emb_cost = split_cosine_dist(det_embs, trks_embs)
            emb_cost = emb_cost.T
        w_assoc_emb = 0.75
        aw_param = 0.4

    matches = []
    if not det_app:
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost *= w_matrix
                else:
                    emb_cost *= w_assoc_emb
                if not mot_off:
                    final_cost = -(iou_matrix + emb_cost)
                    matched_indices = linear_assignment(final_cost)
                else:
                    final_cost = -emb_cost
                    matched_indices = linear_assignment(final_cost)
        else:
            matched_indices = np.empty(shape=(0, 2))
    else:
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    # Filter out those pairs with small IoU
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < threshold:
            unmatched_detections.append(m[1])
            unmatched_trackers.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trackers), np.array(unmatched_detections)
def associate_detections_to_trackers_fusion(
    detections,
    trackers,
    aw_off,
    grid_off,
    mot_off,
    iou_threshold,
    det_embs=None,
    det_app=False,
    appearance_weight=None,
    use_rotated_geom=False,
    risk_options=None,
    return_diagnostics=False,
):
    """
    Baseline 3D association.
    """

    dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []

    trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in trackers]
    trks_confidece = [trk.confidence for trk in trackers]
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)

    diagnostics = {
        'initial_match_count': 0,
        'final_match_count': 0,
        'deferred_pairs': [],
        'deferred_det_indices': [],
        'deferred_trk_indices': [],
    }

    if len(trks_8corner) == 0:
        if return_diagnostics:
            return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0,), dtype=int), diagnostics
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            iou_matrix[d, t] = iou3d(det, trk)[0]
            iou_matrix[d, t] = iou_matrix[d, t] / trks_confidece[t]

    det_centers = None
    trk_centers = None
    try:
        if len(detections) > 0:
            det_centers = np.stack([np.asarray(getattr(det, 'bbox'))[:3] for det in detections], axis=0).astype(np.float32)
        if len(trackers) > 0:
            trk_centers = np.stack([np.asarray(getattr(trk, 'pose'))[:3] for trk in trackers], axis=0).astype(np.float32)
    except Exception:
        det_centers = None
        trk_centers = None

    dist_matrix = None
    if det_centers is not None and trk_centers is not None and det_centers.size > 0 and trk_centers.size > 0:
        diff = det_centers[:, None, :] - trk_centers[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    emb_cost = None
    if not det_app:
        if grid_off:
            if det_embs is not None and det_embs.size > 0:
                d_dim = det_embs.shape[1]
                trk_list = []
                for trk in trackers:
                    e = np.asarray(trk.emb) if hasattr(trk, 'emb') else None
                    if e is None:
                        trk_list.append(np.zeros(d_dim, dtype=np.float32))
                        continue
                    if e.ndim == 2:
                        e = e.mean(axis=0)
                    if e.shape[0] != d_dim:
                        m = min(d_dim, e.shape[0])
                        v = np.zeros(d_dim, dtype=np.float32)
                        v[:m] = e[:m]
                        e = v
                    trk_list.append(e)
                trks_embs = np.vstack(trk_list) if len(trk_list) > 0 else np.zeros((0, d_dim), dtype=np.float32)
                if trks_embs.shape[0] > 0 and det_embs.shape[0] > 0:
                    emb_cost = det_embs @ trks_embs.T
        else:
            trks_embs = np.asarray([trk_emb.emb.tolist() for trk_emb in trackers])
            emb_cost = split_cosine_dist(det_embs, trks_embs)
        w_assoc_emb = 0.75
        aw_param = 0.4

    thr_vec = np.array([
        max(iou_threshold * float(getattr(trk, 'beta_t', 1.0)), 0.02)
        for trk in trackers
    ], dtype=np.float32) if len(trackers) > 0 else None

    base_dist = 3.0
    dist_thr_vec = None
    if dist_matrix is not None and len(trackers) > 0:
        betas = np.array([max(1e-6, float(getattr(trk, 'beta_t', 1.0))) for trk in trackers], dtype=np.float32)
        dist_thr_vec = (base_dist / np.sqrt(betas)).astype(np.float32)

    if min(iou_matrix.shape) > 0:
        candidate_mask = (iou_matrix > thr_vec.reshape(1, -1)).astype(np.int32) if thr_vec is not None else (iou_matrix > iou_threshold).astype(np.int32)
        if dist_thr_vec is not None and dist_matrix is not None:
            candidate_mask = (candidate_mask & (dist_matrix <= dist_thr_vec.reshape(1, -1))).astype(np.int32)

        if candidate_mask.sum(1).max() == 1 and candidate_mask.sum(0).max() == 1:
            matched_indices = np.stack(np.where(candidate_mask), axis=1)
        else:
            score_matrix = iou_matrix.copy()
            if use_rotated_geom and len(detections) > 0 and len(trackers) > 0:
                geom_matrix = np.zeros_like(score_matrix, dtype=np.float32)
                for d_idx, det in enumerate(detections):
                    det_box = getattr(det, 'bbox', None)
                    det_corners = dets_8corner[d_idx]
                    for t_idx, trk in enumerate(trackers):
                        trk_box = getattr(trk, 'pose', None)
                        trk_corners = trks_8corner[t_idx]
                        geom_matrix[d_idx, t_idx] = compute_rotated_ground_similarity(
                            det_box,
                            trk_box,
                            det_corners=det_corners,
                            trk_corners=trk_corners,
                        )
                score_matrix = geom_matrix
            if not det_app and emb_cost is not None and emb_cost.size > 0:
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost = emb_cost * w_matrix
                else:
                    emb_cost = emb_cost * w_assoc_emb
                w_app = min(max(appearance_weight if appearance_weight is not None else 0.10, 0.0), 0.15)
                score_matrix = (1.0 - w_app) * score_matrix + w_app * emb_cost

            pair_risk, identity_support, row_ambiguity, col_ambiguity = compute_association_risk_context(
                score_matrix,
                candidate_mask,
                detections,
                trackers,
                emb_cost=emb_cost,
                risk_options=risk_options,
            )
            score_matrix = apply_association_risk_tempering(
                score_matrix,
                pair_risk,
                identity_support,
                risk_options=risk_options,
            )

            score_matrix[candidate_mask == 0] = -1e9
            if dist_thr_vec is not None and dist_matrix is not None:
                score_matrix = score_matrix.copy()
                score_matrix[dist_matrix > dist_thr_vec.reshape(1, -1)] = -1e9
            matched_indices = linear_assignment(-score_matrix)
            matched_indices, diagnostics = apply_deferred_commitment(
                matched_indices,
                score_matrix,
                candidate_mask,
                pair_risk,
                identity_support,
                row_ambiguity,
                col_ambiguity,
                options=risk_options,
            )
    else:
        matched_indices = np.empty(shape=(0, 2))

    matches = []
    unmatched_detections = []
    unmatched_trackers = []

    for d in range(len(dets_8corner)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    for t in range(len(trks_8corner)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        det_idx, trk_idx = int(m[0]), int(m[1])
        dyn_iou_thr = float(thr_vec[trk_idx]) if thr_vec is not None and trk_idx < thr_vec.size else float(iou_threshold)
        iou_ok = iou_matrix[det_idx, trk_idx] >= dyn_iou_thr
        dist_ok = True
        if dist_thr_vec is not None and dist_matrix is not None and trk_idx < dist_thr_vec.size:
            dist_ok = dist_matrix[det_idx, trk_idx] <= dist_thr_vec[trk_idx]
        if iou_ok and dist_ok:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(det_idx)
            unmatched_trackers.append(trk_idx)

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    if return_diagnostics:
        diagnostics['final_match_count'] = int(matches.shape[0]) if matches.ndim == 2 else 0
        diagnostics['deferred_det_indices'] = sorted(set(int(x) for x in diagnostics.get('deferred_det_indices', [])))
        diagnostics['deferred_trk_indices'] = sorted(set(int(x) for x in diagnostics.get('deferred_trk_indices', [])))
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers), diagnostics
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def trackfusion2Dand3D(tracker_2D, trks_3Dto2D_image, iou_threshold):
    track_indices = list(range(len(tracker_2D)))  # 跟踪对象索引
    detection_indices = list(range(len(trks_3Dto2D_image)))  # 检测对象索引
    matches = []
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    iou_matrix = np.zeros((len(tracker_2D), len(trks_3Dto2D_image)), dtype=np.float32)
    for t, trk in enumerate(tracker_2D):
        for d, det in enumerate(trks_3Dto2D_image):
            iou_matrix[t, d] = iou_batch(trk.x1y1x2y2(), det)  # det: 8 x 3, trk: 8 x 3
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    unmatched_detections = []
    for d, det in enumerate(trks_3Dto2D_image):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    unmatched_tracker_2D = []
    for t, trk in enumerate(tracker_2D):
        if t not in matched_indices[:, 0]:
            unmatched_tracker_2D.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracker_2D.append(m[0])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_tracker_2D), np.array(unmatched_detections)

def associate_2D_to_3D_tracking(tracker_2D, tracks_3D, calib_file, iou_threshold):
	trks_3Dto2D_image = [list(i.additional_info[2:6])  for i in tracks_3D]
	matched_track_2D, unmatch_tracker_2D, _ = trackfusion2Dand3D(tracker_2D, trks_3Dto2D_image, iou_threshold)
	return matched_track_2D, unmatch_tracker_2D


def compute_aw_new_metric(emb_cost, w_association_emb, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus

