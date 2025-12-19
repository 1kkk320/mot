# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance
import scipy.spatial as sp
from tracking.cost_matrix_fusion import compute_fused_cost_matrix
from copy import deepcopy
from tracking.adaptive_angle_weight import compute_adaptive_cost_matrix_weights


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
    threshold = 0.3
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
def associate_detections_to_trackers_fusion(detections, trackers, aw_off, grid_off, mot_off, iou_threshold, det_embs=None, det_app=False, angle_config=None, enable_angle=False, appearance_weight=None):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    将检测分配给跟踪的对象（均表示为边界框）
    detections:  N x 8 x 3
    trackers:    M x 8 x 3
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    dets_8corner = [convert_3dbox_to_8corner(det_tmp.bbox) for det_tmp in detections]
    if len(dets_8corner) > 0:
        dets_8corner = np.stack(dets_8corner, axis=0)
    else:
        dets_8corner = []

    # 当关闭外观或轨迹未包含嵌入时，避免在此处访问 emb；
    # 真正需要时在下面的具体分支中按需构建。
    trks_embs = None
    trks_8corner = [convert_3dbox_to_8corner(trk_tmp.pose) for trk_tmp in trackers]
    trks_confidece = [trk.confidence for trk in trackers]
    if len(trks_8corner) > 0:
        trks_8corner = np.stack(trks_8corner, axis=0)
    if (len(trks_8corner)==0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    # iou_matrix_emb = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    # eucliDistance_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    # 计算运动特征
    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
            # print("更新前", iou_matrix[d, t])
            iou_matrix[d, t] = iou_matrix[d, t]/trks_confidece[t]  # 除以相应的预测置信度，得到新的关联矩阵
            # print("更新后",iou_matrix[d, t],type(iou_matrix[d, t]),"置信度",trks_confidece[t])
            # if np.isinf(iou_matrix[d, t]) or np.isnan(iou_matrix[d, t]):
            # 	iou_matrix[d, t] = 0

    if not det_app:
        if grid_off:
            if det_embs is None or det_embs.size == 0:
                emb_cost = None
            else:
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
                emb_cost = None if (trks_embs.shape[0] == 0 or det_embs.shape[0] == 0) else det_embs @ trks_embs.T
        else:
            trks_embs = np.asarray([trk_emb.emb.tolist() for trk_emb in trackers])
            emb_cost = split_cosine_dist(det_embs, trks_embs)
        w_assoc_emb = 0.75
        aw_param = 0.4

    matches = []
    if min(iou_matrix.shape) > 0:
        # 可选的快速唯一匹配分支（基于IoU阈值）
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # 自适应外观加权
            app_matrix_det_trk = None
            if not det_app and emb_cost is not None and emb_cost.size > 0:
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost = emb_cost * w_matrix
                else:
                    emb_cost = emb_cost * w_assoc_emb
                app_matrix_det_trk = emb_cost

            w_app = 0.0
            appearance_reliable = False
            if (not det_app) and (app_matrix_det_trk is not None) and (app_matrix_det_trk.size > 0):
                A = app_matrix_det_trk
                row_ratio = 0.0
                col_ratio = 0.0
                # 放宽门控：margin>=0.07, top1>=0.45
                margin_thr = 0.07
                top1_thr = 0.45
                if A.shape[1] >= 2:
                    sr = np.sort(A, axis=1)
                    top1 = sr[:, -1]
                    top2 = sr[:, -2]
                    row_ok = ((top1 - top2) >= margin_thr) & (top1 >= top1_thr)
                    row_ratio = row_ok.mean() if row_ok.size > 0 else 0.0
                B = A.T
                if B.shape[1] >= 2:
                    sc = np.sort(B, axis=1)
                    t1 = sc[:, -1]
                    t2 = sc[:, -2]
                    col_ok = ((t1 - t2) >= margin_thr) & (t1 >= top1_thr)
                    col_ratio = col_ok.mean() if col_ok.size > 0 else 0.0
                # 放宽比例阈值
                appearance_reliable = (row_ratio >= 0.25 and col_ratio >= 0.25) or (row_ratio >= 0.40 or col_ratio >= 0.40)

                # 计算基础外观权重（受 appearance_weight 控制，最大0.15）
                base_app = appearance_weight if (appearance_weight is not None) else 0.10
                if base_app > 0.15:
                    base_app = 0.15
                if base_app < 0.0:
                    base_app = 0.0

                if appearance_reliable:
                    w_app = base_app
                else:
                    # 保底权重：仅在用户未显式设为0且确有嵌入时给极小权重
                    w_app_min = 0.05
                    w_app = min(w_app_min, base_app)
            angle_cfg_for_call = angle_config
            if angle_config is not None:
                try:
                    angle_cfg_for_call = deepcopy(angle_config)
                except Exception:
                    angle_cfg_for_call = angle_config
                if hasattr(angle_cfg_for_call, 'enable_angle_feature'):
                    angle_cfg_for_call.enable_angle_feature = True
                if hasattr(angle_cfg_for_call, 'angle_cost_sigma'):
                    angle_cfg_for_call.angle_cost_sigma = 0.35
                if hasattr(angle_cfg_for_call, 'angle_weight'):
                    angle_cfg_for_call.angle_weight = 0.25
            residual = max(0.0, 1.0 - w_app)
            w_iou = residual * 0.75
            w_ang = residual * 0.25
            weights = {
                'iou': w_iou,
                'velocity': 0.0,
                'appearance': w_app,
                'angle': w_ang  # base angle weight (will be scaled per-pair if enabled)
            }
            total_w = w_app + w_iou + w_ang

            # Gaussian adaptive angle weights (pairwise): scale angle weight per (track, det)
            # Only applied when angle feature is enabled
            angle_weight_matrix = None
            if enable_angle:
                # Build angle arrays for trackers/detections
                def _angle_from_track(trk):
                    try:
                        if hasattr(trk, 'pose') and trk.pose is not None and len(trk.pose) >= 7:
                            return float(trk.pose[6])
                        if hasattr(trk, 'angle'):
                            return float(trk.angle)
                        if hasattr(trk, 'bbox') and trk.bbox is not None:
                            if len(trk.bbox) >= 7:
                                return float(trk.bbox[6])
                            if len(trk.bbox) >= 5:
                                return float(trk.bbox[4])
                    except Exception:
                        pass
                    return 0.0

                def _angle_from_det(det):
                    try:
                        if hasattr(det, 'bbox') and det.bbox is not None:
                            if len(det.bbox) >= 7:
                                return float(det.bbox[6])
                            if len(det.bbox) >= 5:
                                return float(det.bbox[4])
                        if hasattr(det, 'angle'):
                            return float(det.angle)
                        if isinstance(det, (list, tuple, np.ndarray)):
                            if len(det) >= 7:
                                return float(det[6])
                            if len(det) >= 5:
                                return float(det[4])
                    except Exception:
                        pass
                    return 0.0

                track_angles = np.array([_angle_from_track(t) for t in trackers], dtype=np.float32)
                det_angles = np.array([_angle_from_det(d) for d in detections], dtype=np.float32)

                if track_angles.size > 0 and det_angles.size > 0:
                    # Use gaussian method to produce unit angle weights in [0,1]
                    base_w = {'iou': 0.0, 'velocity': 0.0, 'appearance': 0.0, 'angle': 1.0}
                    adaptive_w = compute_adaptive_cost_matrix_weights(
                        track_angles, det_angles,
                        base_weights=base_w,
                        angle_weight_method='gaussian',
                        verbose=False
                    )
                    # Scale by base scalar w_ang to get final per-pair angle weights
                    angle_weight_matrix = w_ang * adaptive_w['angle']  # shape: (trks, dets)


            fused_cost, angle_cost, gate_mask = compute_fused_cost_matrix(
                trackers,
                detections,
                iou_matrix=iou_matrix.T,  # 转为 (trks x dets)
                velocity_matrix=None,
                appearance_matrix=(app_matrix_det_trk.T if (not det_app and app_matrix_det_trk is not None) else None),
                angle_config=(angle_cfg_for_call if enable_angle else None),
                weights={
                    'iou': w_iou,
                    'velocity': 0.0,
                    'appearance': w_app,
                    # If per-pair weights exist, pass matrix; otherwise fall back to scalar
                    'angle': (angle_weight_matrix if (angle_weight_matrix is not None) else w_ang)
                },
                verbose=False
            )

            # 赋值为 (dets x trks) 供 linear_assignment 使用
            final_cost = fused_cost.T
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(dets_8corner):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trks_8corner):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
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

