# -*-coding:utf-8-*
# author: wangxy
import numpy as np
import math
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance
import scipy.spatial as sp


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def extract_track_heading(track):
    try:
        if hasattr(track, 'pose') and track.pose is not None and len(track.pose) >= 4:
            return float(track.pose[3])
    except Exception:
        pass
    return None


def extract_detection_heading(detection):
    try:
        if hasattr(detection, 'bbox') and detection.bbox is not None and len(detection.bbox) >= 4:
            return float(detection.bbox[3])
    except Exception:
        pass
    return None


def compute_heading_distance(track_heading, det_heading, metric='symmetric'):
    if track_heading is None or det_heading is None:
        return 0.0

    delta = wrap_to_pi(track_heading - det_heading)

    if metric == 'cosine':
        return 0.5 * (1.0 - math.cos(delta))

    # symmetric: theta and theta + pi are treated as equivalent
    symmetric_delta = math.acos(np.clip(abs(math.cos(delta)), 0.0, 1.0))
    return symmetric_delta / (math.pi / 2.0)


def _update_heading_stats(stats, **kwargs):
    if stats is None:
        return
    for key, value in kwargs.items():
        stats[key] = stats.get(key, 0) + value


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
def associate_detections_to_trackers_fusion(detections, trackers, aw_off, grid_off, mot_off, iou_threshold, det_embs=None, det_app=False, appearance_weight=None, heading_options=None):
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

    if len(trks_8corner) == 0:
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

    heading_options = heading_options or {}
    heading_rescore_enabled = bool(heading_options.get('enabled', False))
    heading_enabled = heading_rescore_enabled or bool(heading_options.get('pre_gate_enabled', False))
    heading_metric = heading_options.get('metric', 'symmetric')
    heading_weight = float(heading_options.get('weight', 0.20))
    heading_margin = float(heading_options.get('ambiguity_margin', 0.03))
    heading_pre_gate_enabled = bool(heading_options.get('pre_gate_enabled', False))
    heading_pre_gate_threshold = float(heading_options.get('pre_gate_threshold', 0.55))
    heading_hard_gate_threshold = float(heading_options.get('hard_gate_threshold', 0.45))
    heading_stats = heading_options.get('stats')

    if heading_enabled:
        _update_heading_stats(heading_stats, calls=1)

    if min(iou_matrix.shape) > 0:
        candidate_mask = (iou_matrix > thr_vec.reshape(1, -1)).astype(np.int32) if thr_vec is not None else (iou_matrix > iou_threshold).astype(np.int32)
        if dist_thr_vec is not None and dist_matrix is not None:
            candidate_mask = (candidate_mask & (dist_matrix <= dist_thr_vec.reshape(1, -1))).astype(np.int32)
        baseline_candidate_mask = candidate_mask.copy()

        if heading_enabled and heading_pre_gate_enabled and candidate_mask.any():
            pre_gate_mask = np.zeros_like(candidate_mask, dtype=bool)
            for det_idx, trk_idx in zip(*np.where(candidate_mask > 0)):
                det_heading = extract_detection_heading(detections[det_idx])
                trk_heading = extract_track_heading(trackers[trk_idx])
                heading_dist = compute_heading_distance(
                    trk_heading,
                    det_heading,
                    metric=heading_metric
                )
                if heading_dist >= heading_pre_gate_threshold:
                    pre_gate_mask[det_idx, trk_idx] = True
            if pre_gate_mask.any():
                candidate_mask = candidate_mask.copy()
                candidate_mask[pre_gate_mask] = 0
                _update_heading_stats(
                    heading_stats,
                    pre_gate_suppressed_pairs=int(pre_gate_mask.sum())
                )

        if candidate_mask.sum(1).max() == 1 and candidate_mask.sum(0).max() == 1:
            matched_indices = np.stack(np.where(candidate_mask), axis=1)
        else:
            score_matrix = iou_matrix.copy()
            if not det_app and emb_cost is not None and emb_cost.size > 0:
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost = emb_cost * w_matrix
                else:
                    emb_cost = emb_cost * w_assoc_emb
                w_app = min(max(appearance_weight if appearance_weight is not None else 0.10, 0.0), 0.15)
                score_matrix = (1.0 - w_app) * score_matrix + w_app * emb_cost

            baseline_score_matrix = score_matrix.copy()
            baseline_gated_score_matrix = baseline_score_matrix.copy()
            if dist_thr_vec is not None and dist_matrix is not None:
                baseline_gated_score_matrix[dist_matrix > dist_thr_vec.reshape(1, -1)] = -1e9
            baseline_gated_score_matrix[baseline_candidate_mask == 0] = -1e9

            if heading_rescore_enabled:
                ambiguous_mask = np.zeros_like(candidate_mask, dtype=bool)

                for det_idx in range(candidate_mask.shape[0]):
                    valid_cols = np.where(candidate_mask[det_idx] > 0)[0]
                    if valid_cols.size >= 2:
                        row_scores = score_matrix[det_idx, valid_cols]
                        best_score = row_scores.max()
                        near_best = valid_cols[(best_score - row_scores) <= heading_margin]
                        if near_best.size >= 2:
                            ambiguous_mask[det_idx, near_best] = True

                for trk_idx in range(candidate_mask.shape[1]):
                    valid_rows = np.where(candidate_mask[:, trk_idx] > 0)[0]
                    if valid_rows.size >= 2:
                        col_scores = score_matrix[valid_rows, trk_idx]
                        best_score = col_scores.max()
                        near_best = valid_rows[(best_score - col_scores) <= heading_margin]
                        if near_best.size >= 2:
                            ambiguous_mask[near_best, trk_idx] = True

                if ambiguous_mask.any():
                    _update_heading_stats(
                        heading_stats,
                        ambiguous_calls=1,
                        ambiguous_pairs=int(ambiguous_mask.sum())
                    )
                    heading_penalty = np.zeros_like(score_matrix, dtype=np.float32)
                    for det_idx, trk_idx in zip(*np.where(ambiguous_mask)):
                        det_heading = extract_detection_heading(detections[det_idx])
                        trk_heading = extract_track_heading(trackers[trk_idx])
                        heading_penalty[det_idx, trk_idx] = compute_heading_distance(
                            trk_heading,
                            det_heading,
                            metric=heading_metric
                        )
                    suppress_mask = ambiguous_mask & (heading_penalty >= heading_hard_gate_threshold)
                    if suppress_mask.any():
                        score_matrix = score_matrix.copy()
                        score_matrix[suppress_mask] = -1e9
                        _update_heading_stats(
                            heading_stats,
                            suppressed_pairs=int(suppress_mask.sum())
                        )
                    soft_mask = ambiguous_mask & (~suppress_mask)
                    if soft_mask.any():
                        score_matrix = score_matrix.copy()
                        score_matrix[soft_mask] = score_matrix[soft_mask] - heading_weight * heading_penalty[soft_mask]

            score_matrix[candidate_mask == 0] = -1e9
            if dist_thr_vec is not None and dist_matrix is not None:
                score_matrix = score_matrix.copy()
                score_matrix[dist_matrix > dist_thr_vec.reshape(1, -1)] = -1e9

            if heading_rescore_enabled:
                valid_row_mask = np.any(baseline_gated_score_matrix > -1e8, axis=1)
                valid_col_mask = np.any(baseline_gated_score_matrix > -1e8, axis=0)
                if valid_row_mask.any():
                    baseline_row_best = np.argmax(baseline_gated_score_matrix, axis=1)
                    rescored_row_best = np.argmax(score_matrix, axis=1)
                    row_flips = int(np.sum((baseline_row_best != rescored_row_best) & valid_row_mask))
                    if row_flips > 0:
                        _update_heading_stats(heading_stats, winner_flip_rows=row_flips)
                if valid_col_mask.any():
                    baseline_col_best = np.argmax(baseline_gated_score_matrix, axis=0)
                    rescored_col_best = np.argmax(score_matrix, axis=0)
                    col_flips = int(np.sum((baseline_col_best != rescored_col_best) & valid_col_mask))
                    if col_flips > 0:
                        _update_heading_stats(heading_stats, winner_flip_cols=col_flips)
                baseline_matched_indices = linear_assignment(-baseline_gated_score_matrix)
            else:
                baseline_matched_indices = linear_assignment(-baseline_gated_score_matrix)
            matched_indices = linear_assignment(-score_matrix)

            if heading_enabled:
                if heading_pre_gate_enabled:
                    baseline_candidate_pairs = set((int(r), int(c)) for r, c in zip(*np.where(baseline_candidate_mask > 0)))
                    gated_candidate_pairs = set((int(r), int(c)) for r, c in zip(*np.where(candidate_mask > 0)))
                    removed_pairs = baseline_candidate_pairs - gated_candidate_pairs
                    if removed_pairs:
                        baseline_pre_gate_matrix = baseline_score_matrix.copy()
                        baseline_pre_gate_matrix[baseline_candidate_mask == 0] = -1e9
                        if dist_thr_vec is not None and dist_matrix is not None:
                            baseline_pre_gate_matrix[dist_matrix > dist_thr_vec.reshape(1, -1)] = -1e9
                        pre_gate_matrix = baseline_score_matrix.copy()
                        pre_gate_matrix[candidate_mask == 0] = -1e9
                        if dist_thr_vec is not None and dist_matrix is not None:
                            pre_gate_matrix[dist_matrix > dist_thr_vec.reshape(1, -1)] = -1e9
                        baseline_pre_gate_matches = linear_assignment(-baseline_pre_gate_matrix)
                        post_pre_gate_matches = linear_assignment(-pre_gate_matrix)
                        baseline_pre_gate_pairs = set((int(m[0]), int(m[1])) for m in baseline_pre_gate_matches)
                        post_pre_gate_pairs = set((int(m[0]), int(m[1])) for m in post_pre_gate_matches)
                        if baseline_pre_gate_pairs != post_pre_gate_pairs:
                            _update_heading_stats(
                                heading_stats,
                                pre_gate_changed_matches=len(baseline_pre_gate_pairs.symmetric_difference(post_pre_gate_pairs))
                            )
                baseline_pairs = set((int(m[0]), int(m[1])) for m in baseline_matched_indices)
                rescored_pairs = set((int(m[0]), int(m[1])) for m in matched_indices)
                if baseline_pairs != rescored_pairs:
                    changed_pairs = len(baseline_pairs.symmetric_difference(rescored_pairs))
                    _update_heading_stats(
                        heading_stats,
                        changed_calls=1,
                        changed_pairs=changed_pairs
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

