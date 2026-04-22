# -*-coding:utf-8-*
# author: wangxy
import numpy as np
import math
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance, get_velocity
import scipy.spatial as sp
from tracking.cost_matrix_fusion import compute_fused_cost_matrix
from copy import deepcopy
from tracking.adaptive_angle_weight import compute_adaptive_cost_matrix_weights
from tracking.angle_feature import compute_angle_similarity_matrix, compute_speed_adaptive_sigma, angle_gate


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
    # L1 概览与空候选提示（受 angle_config.verbose 控制）
    verbose_flag = bool(getattr(angle_config, 'verbose', False))
    if verbose_flag:
        en_ang = bool(enable_angle and (angle_config is not None) and getattr(angle_config, 'enable_angle_feature', False))
        try:
            ang_w0 = float(getattr(angle_config, 'angle_weight', 0.0)) if angle_config is not None else 0.0
        except Exception:
            ang_w0 = 0.0
        method0 = getattr(angle_config, 'angle_cost_method', None) if angle_config is not None else None
        try:
            print(f"[L1 概览] dets={len(dets_8corner)} trks={len(trks_8corner)} enable_angle={en_ang} method={method0} angle_w={ang_w0:.3f}", flush=True)
        except Exception:
            pass
        if len(dets_8corner) == 0 or len(trks_8corner) == 0:
            try:
                print(f"[L1 提示] 空候选: 检测={len(dets_8corner)}, 轨迹={len(trks_8corner)}", flush=True)
            except Exception:
                pass
    if (len(trks_8corner)==0):
        return np.empty((0, 2), dtype=int), np.arange(len(dets_8corner)), np.empty((0, 8, 3), dtype=int)

    iou_matrix = np.zeros((len(dets_8corner), len(trks_8corner)), dtype=np.float32)
    # 计算运动特征
    for d, det in enumerate(dets_8corner):
        for t, trk in enumerate(trks_8corner):
            iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
            iou_matrix[d, t] = iou_matrix[d, t]/trks_confidece[t]  # 除以相应的预测置信度，得到新的关联矩阵
    # ========== 距离门控（优先）: 以中心点欧氏距离为门控 ==========
    # 检测与轨迹中心
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
        # (dets x trks)
        diff = det_centers[:, None, :] - trk_centers[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)

    def _angle_from_track_for_gate(trk):
        try:
            if hasattr(trk, 'angle_smoothed') and trk.angle_smoothed is not None:
                return float(trk.angle_smoothed)
            if hasattr(trk, 'pose') and trk.pose is not None and len(trk.pose) >= 7:
                return float(trk.pose[3])
            if hasattr(trk, 'angle'):
                return float(trk.angle)
            if hasattr(trk, 'bbox') and trk.bbox is not None:
                if len(trk.bbox) >= 7:
                    return float(trk.bbox[3])
                if len(trk.bbox) >= 5:
                    return float(trk.bbox[4])
        except Exception:
            pass
        return 0.0

    def _angle_from_det_for_gate(det):
        try:
            if hasattr(det, 'bbox') and det.bbox is not None:
                if len(det.bbox) >= 7:
                    return float(det.bbox[3])
                if len(det.bbox) >= 5:
                    return float(det.bbox[4])
            if hasattr(det, 'angle'):
                return float(det.angle)
            if isinstance(det, (list, tuple, np.ndarray)):
                if len(det) >= 7:
                    return float(det[3])
                if len(det) >= 5:
                    return float(det[4])
        except Exception:
            pass
        return 0.0

    def _build_ambiguity_pair_mask(candidate_mask, score_matrix=None, min_candidates=2, gap_threshold=None):
        try:
            cand = np.asarray(candidate_mask, dtype=np.int32)
            if cand.ndim != 2 or cand.size == 0:
                return None
            min_c = max(2, int(min_candidates))
            row_counts = cand.sum(axis=1, keepdims=True)
            col_counts = cand.sum(axis=0, keepdims=True)
            ambiguous = ((row_counts >= min_c) | (col_counts >= min_c)).astype(np.int32)

            if score_matrix is not None and gap_threshold is not None:
                scores = np.asarray(score_matrix, dtype=np.float32)
                gap_thr = max(0.0, float(gap_threshold))
                row_gap_mask = np.zeros_like(cand, dtype=bool)
                col_gap_mask = np.zeros_like(cand, dtype=bool)

                for det_idx in range(cand.shape[0]):
                    valid_cols = np.where(cand[det_idx] > 0)[0]
                    if valid_cols.size >= 2:
                        vals = scores[det_idx, valid_cols]
                        order = np.sort(vals)[::-1]
                        if (order[0] - order[1]) <= gap_thr:
                            row_gap_mask[det_idx, valid_cols] = True

                for trk_idx in range(cand.shape[1]):
                    valid_rows = np.where(cand[:, trk_idx] > 0)[0]
                    if valid_rows.size >= 2:
                        vals = scores[valid_rows, trk_idx]
                        order = np.sort(vals)[::-1]
                        if (order[0] - order[1]) <= gap_thr:
                            col_gap_mask[valid_rows, trk_idx] = True

                ambiguous = ambiguous & (row_gap_mask | col_gap_mask)
            return (cand > 0) & (ambiguous > 0)
        except Exception:
            return None

    def _motion_heading_similarity_matrix(track_angles, det_angles, tracks, sigma=0.35):
        n_trks = len(tracks)
        n_dets = len(det_angles)
        sim = np.ones((n_trks, n_dets), dtype=np.float32)
        if n_trks == 0 or n_dets == 0:
            return sim

        def _yaw_from_velocity(trk):
            try:
                vel = np.asarray(get_velocity(trk)).reshape(-1)
                if vel.size >= 3:
                    vx = float(vel[0])
                    vz = float(vel[2])
                    speed_xz = math.hypot(vx, vz)
                    if speed_xz > 1e-3:
                        return math.atan2(vx, vz)
            except Exception:
                pass
            return None

        sigma_val = max(1e-3, float(sigma))
        for trk_idx, trk in enumerate(tracks):
            motion_yaw = _yaw_from_velocity(trk)
            if motion_yaw is None:
                continue
            track_delta = compute_angle_similarity_matrix(
                np.array([track_angles[trk_idx]], dtype=np.float32),
                np.array([motion_yaw], dtype=np.float32),
                method='symmetric',
                sigma=sigma_val,
                gate_threshold=None,
            )[0][0, 0]
            for det_idx in range(n_dets):
                det_delta = compute_angle_similarity_matrix(
                    np.array([det_angles[det_idx]], dtype=np.float32),
                    np.array([motion_yaw], dtype=np.float32),
                    method='symmetric',
                    sigma=sigma_val,
                    gate_threshold=None,
                )[0][0, 0]
                sim[trk_idx, det_idx] = np.float32(max(0.0, 1.0 - 0.5 * (track_delta + det_delta)))
        return sim

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
    thr_vec = None  # IoU动态阈值
    if len(trks_8corner) > 0:
        thr_vec = np.array([
            max(iou_threshold * float(getattr(trk, 'beta_t', 1.0)), 0.02)
            for trk in trackers
        ], dtype=np.float32)
    # 距离动态阈值: base_dist / sqrt(beta_t)
    # 说明: base_dist 为基础距离阈值，可按需要调整。未提供外部参数时使用默认值。
    base_dist = 3.0
    dist_thr_vec = None
    if dist_matrix is not None and len(trackers) > 0:
        betas = np.array([max(1e-6, float(getattr(trk, 'beta_t', 1.0))) for trk in trackers], dtype=np.float32)
        dist_thr_vec = (base_dist / np.sqrt(betas)).astype(np.float32)

    if min(iou_matrix.shape) > 0:
        if thr_vec is not None and thr_vec.size == iou_matrix.shape[1]:
            a = (iou_matrix > thr_vec.reshape(1, -1)).astype(np.int32)
        else:
            a = (iou_matrix > iou_threshold).astype(np.int32)
        # 距离门控优先：在唯一匹配分支前先按距离阈值过滤候选
        if dist_thr_vec is not None and dist_thr_vec.size == iou_matrix.shape[1] and dist_matrix is not None:
            a = (a & (dist_matrix <= dist_thr_vec.reshape(1, -1))).astype(np.int32)
        
        # ========== 身份冲突校验（Soft Verification）==========
        # 目的：利用角度作为"纠错器"而非"过滤器"
        # 策略：角度差异过大时，将匹配降级到fused_cost路径重新竞争
        enable_angle_conflict_check = False  # 默认关闭
        if hasattr(angle_config, 'enable_angle_conflict_check'):
            enable_angle_conflict_check = bool(angle_config.enable_angle_conflict_check)
        
        if enable_angle and enable_angle_conflict_check and angle_config is not None:
            # 获取身份冲突阈值（默认45度）
            conflict_angle_threshold = math.radians(45)
            if hasattr(angle_config, 'conflict_angle_threshold'):
                conflict_angle_threshold = float(angle_config.conflict_angle_threshold)
            
            # 角度归一化函数：将角度归一化到[-π, π]
            def normalize_angle(angle):
                """将角度归一化到[-π, π]范围"""
                while angle > math.pi:
                    angle -= 2 * math.pi
                while angle < -math.pi:
                    angle += 2 * math.pi
                return angle
            
            # 对称性感知角度差计算
            def compute_symmetric_angle_diff(angle1, angle2):
                """
                计算对称性感知的角度差
                使用公式: Δθ_sym = arccos(|cos(Δθ)|)
                效果: 0°和180°被视为相同（角度差=0°）
                """
                delta = normalize_angle(angle1 - angle2)
                cos_delta = abs(math.cos(delta))
                delta_sym = math.acos(np.clip(cos_delta, 0.0, 1.0))
                return delta_sym
            
            # 提取角度
            def _angle_from_track(trk):
                try:
                    if hasattr(trk, 'angle_smoothed') and trk.angle_smoothed is not None:
                        return normalize_angle(float(trk.angle_smoothed))
                    if hasattr(trk, 'pose') and trk.pose is not None and len(trk.pose) >= 7:
                        return normalize_angle(float(trk.pose[3]))
                except Exception:
                    pass
                return 0.0
            
            def _angle_from_det(det):
                try:
                    if hasattr(det, 'bbox') and det.bbox is not None and len(det.bbox) >= 7:
                        return normalize_angle(float(det.bbox[3]))
                except Exception:
                    pass
                return 0.0
            
            track_angles = np.array([_angle_from_track(t) for t in trackers], dtype=np.float32)
            det_angles = np.array([_angle_from_det(d) for d in detections], dtype=np.float32)
            
            if track_angles.size > 0 and det_angles.size > 0:
                # 计算对称角度差异矩阵
                angle_diff_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
                for d in range(len(detections)):
                    for t in range(len(trackers)):
                        # 使用对称性感知计算
                        angle_diff_matrix[d, t] = compute_symmetric_angle_diff(
                            det_angles[d], track_angles[t]
                        )
                
                # 身份冲突检测：对称角度差异>阈值 → 标记为冲突（但不直接拒绝）
                conflict_mask = (angle_diff_matrix > conflict_angle_threshold).astype(np.int32)
                
                # 统计冲突候选
                n_conflicts = int((a & conflict_mask).sum())
                
                # ========== 详细诊断：分析冲突匹配 ==========
                if verbose_flag and n_conflicts > 0:
                    conflict_pairs = np.where(a & conflict_mask)
                    print(f"\n[身份冲突校验] 检测到 {n_conflicts} 个潜在冲突", flush=True)
                    print(f"{'Det':<4} {'Trk':<4} {'IoU':<6} {'对称角度差':<12} {'检测角度':<10} {'轨迹角度':<10} {'处理方式'}", flush=True)
                    print("-" * 80, flush=True)
                    
                    for d, t in zip(conflict_pairs[0], conflict_pairs[1]):
                        iou_val = iou_matrix[d, t]
                        angle_diff_sym_deg = math.degrees(angle_diff_matrix[d, t])
                        det_angle_deg = math.degrees(det_angles[d])
                        trk_angle_deg = math.degrees(track_angles[t])
                        
                        print(f"{d:<4} {t:<4} {iou_val:<6.3f} {angle_diff_sym_deg:<12.1f} {det_angle_deg:<10.1f} {trk_angle_deg:<10.1f} 降级到fused_cost", flush=True)
                    print("-" * 80 + "\n", flush=True)
                
                # 软校验：将冲突匹配降级到fused_cost路径（破坏unique条件）
                # 这样它们会在fused_cost中与角度+外观+IoU重新竞争
                a = (a & (~conflict_mask)).astype(np.int32)

        if enable_angle and angle_config is not None and getattr(angle_config, 'enable_unique_iou_angle_gate', False):
            unique_iou_angle_threshold = float(
                getattr(
                    angle_config,
                    'unique_iou_angle_threshold',
                    getattr(angle_config, 'angle_gate_threshold', math.pi / 2),
                )
            )
            unique_iou_method = getattr(angle_config, 'angle_cost_method', 'gaussian')
            for det_idx, trk_idx in np.argwhere(a > 0):
                det_angle = _angle_from_det_for_gate(detections[det_idx])
                trk_angle = _angle_from_track_for_gate(trackers[trk_idx])
                if not angle_gate(
                    trk_angle,
                    det_angle,
                    threshold=unique_iou_angle_threshold,
                    method=unique_iou_method,
                ):
                    a[det_idx, trk_idx] = 0

        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
            if verbose_flag:
                try:
                    print(f"[L1 路径] unique_iou: matches={matched_indices.shape[0]}", flush=True)
                except Exception:
                    pass
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
            ambiguity_triggered_angle = False
            ambiguity_pair_mask_det_trk = None
            if enable_angle and angle_cfg_for_call is not None:
                ambiguity_triggered_angle = bool(
                    getattr(angle_cfg_for_call, 'enable_ambiguity_triggered_angle', False)
                )
                if ambiguity_triggered_angle:
                    ambiguity_gap_threshold = None
                    if bool(getattr(angle_cfg_for_call, 'enable_ambiguity_gap_check', False)):
                        ambiguity_gap_threshold = getattr(
                            angle_cfg_for_call,
                            'ambiguity_iou_gap_threshold',
                            0.05
                        )
                    ambiguity_pair_mask_det_trk = _build_ambiguity_pair_mask(
                        a,
                        score_matrix=iou_matrix,
                        min_candidates=getattr(angle_cfg_for_call, 'ambiguity_min_candidates', 2),
                        gap_threshold=ambiguity_gap_threshold
                    )
                    if ambiguity_pair_mask_det_trk is None or not np.any(ambiguity_pair_mask_det_trk):
                        ambiguity_triggered_angle = False
            residual = max(0.0, 1.0 - w_app)
            
            # ========== 路径感知的权重分配 ==========
            # 核心思想：fused_cost路径说明IoU不可靠，需要大幅增加角度权重
            # unique_iou路径（90%）：IoU可靠，不需要角度
            # fused_cost路径（10%）：IoU不可靠，迫切需要角度辅助
            
            # 从 Tracker.angle_level1_weight 传入的 angle_config.angle_weight 读取角度占比（0~1），默认0.25
            ang_share = 0.25
            try:
                if angle_cfg_for_call is not None and hasattr(angle_cfg_for_call, 'angle_weight'):
                    ang_share = float(angle_cfg_for_call.angle_weight)
            except Exception:
                ang_share = 0.25
            if ang_share < 0.0:
                ang_share = 0.0
            if ang_share > 1.0:
                ang_share = 1.0
            
            # ✅ 路径感知增强：在fused_cost路径中，IoU不可靠，大幅增加角度权重
            # 检查是否启用路径感知增强
            enable_path_aware_weighting = True  # 默认启用
            if hasattr(angle_cfg_for_call, 'enable_path_aware_weighting'):
                enable_path_aware_weighting = bool(angle_cfg_for_call.enable_path_aware_weighting)
            
            if enable_path_aware_weighting and not ambiguity_triggered_angle:
                # fused_cost路径说明IoU特征不够可靠（存在多个候选）
                # 需要大幅增加角度权重来辅助区分
                fused_cost_angle_boost = 1.8  # 增强系数（默认1.8倍）
                if hasattr(angle_cfg_for_call, 'fused_cost_angle_boost'):
                    fused_cost_angle_boost = float(angle_cfg_for_call.fused_cost_angle_boost)
                
                ang_share_boosted = min(ang_share * fused_cost_angle_boost, 0.95)  # 最高95%
                
                if verbose_flag:
                    print(f"[路径感知] fused_cost路径，角度权重增强: {ang_share:.3f} → {ang_share_boosted:.3f} (x{fused_cost_angle_boost:.1f})", flush=True)
                
                ang_share = ang_share_boosted
            
            w_ang = residual * ang_share
            w_iou = residual - w_ang
            if verbose_flag:
                try:
                    print(f"[L1 概览] weights: w_app={w_app:.3f}, w_ang={w_ang:.3f}, w_iou={w_iou:.3f}, app_rel={appearance_reliable}", flush=True)
                except Exception:
                    pass
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
                # ========== 角度质量评估 ==========
                from tracking.angle_quality import compute_angle_quality_matrix
                
                # 计算质量矩阵 [n_dets, n_tracks]
                quality_matrix = compute_angle_quality_matrix(detections, trackers)
                
                # ========== 调试：统计质量分布 ==========
                if verbose_flag:
                    n_total = quality_matrix.size
                    n_low_quality = (quality_matrix < 0.5).sum()
                    n_zero_quality = (quality_matrix == 0.0).sum()
                    avg_quality = quality_matrix.mean() if n_total > 0 else 0.0
                    print(f"[角度质量] 总对数={n_total}, 低质量(<0.5)={n_low_quality} ({n_low_quality/n_total*100:.1f}%), "
                          f"零质量(=0.0)={n_zero_quality} ({n_zero_quality/n_total*100:.1f}%), 平均质量={avg_quality:.3f}", flush=True)
                    
                    # ========== 调试：检查权重矩阵 ==========
                    if hasattr(locals(), 'angle_weight_matrix') and angle_weight_matrix is not None:
                        print(f"[权重调试] angle_weight_matrix形状={angle_weight_matrix.shape}, "
                              f"最小值={angle_weight_matrix.min():.3f}, 最大值={angle_weight_matrix.max():.3f}, "
                              f"平均值={angle_weight_matrix.mean():.3f}", flush=True)
                
                # Build angle arrays for trackers/detections
                def _angle_from_track(trk):
                    try:
                        # 优先使用平滑角度（EMA）
                        if hasattr(trk, 'angle_smoothed') and trk.angle_smoothed is not None:
                            return float(trk.angle_smoothed)
                        
                        # 否则使用原始角度
                        # 轨迹pose格式: [x, y, z, rot_y, l, w, h]
                        if hasattr(trk, 'pose') and trk.pose is not None and len(trk.pose) >= 7:
                            return float(trk.pose[3])
                        if hasattr(trk, 'angle'):
                            return float(trk.angle)
                        if hasattr(trk, 'bbox') and trk.bbox is not None:
                            if len(trk.bbox) >= 7:
                                return float(trk.bbox[3])
                            if len(trk.bbox) >= 5:
                                return float(trk.bbox[4])
                    except Exception:
                        pass
                    return 0.0

                def _angle_from_det(det):
                    try:
                        # 检测bbox格式: [x, y, z, rot_y, l, w, h]
                        if hasattr(det, 'bbox') and det.bbox is not None:
                            if len(det.bbox) >= 7:
                                return float(det.bbox[3])
                            if len(det.bbox) >= 5:
                                return float(det.bbox[4])
                        if hasattr(det, 'angle'):
                            return float(det.angle)
                        if isinstance(det, (list, tuple, np.ndarray)):
                            if len(det) >= 7:
                                return float(det[3])
                            if len(det) >= 5:
                                return float(det[4])
                    except Exception:
                        pass
                    return 0.0

                track_angles = np.array([_angle_from_track(t) for t in trackers], dtype=np.float32)
                det_angles = np.array([_angle_from_det(d) for d in detections], dtype=np.float32)

                if track_angles.size > 0 and det_angles.size > 0:
                    use_heading_motion_consistency = bool(
                        ambiguity_triggered_angle and
                        angle_cfg_for_call is not None and
                        getattr(angle_cfg_for_call, 'enable_heading_motion_consistency', False)
                    )
                    # Use gaussian method to produce unit angle weights in [0,1]
                    base_w = {'iou': 0.0, 'velocity': 0.0, 'appearance': 0.0, 'angle': 1.0}
                    
                    # 从angle_config中获取sigma值
                    angle_sigma = None
                    if angle_cfg_for_call is not None and hasattr(angle_cfg_for_call, 'angle_cost_sigma'):
                        angle_sigma = float(angle_cfg_for_call.angle_cost_sigma)
                    if angle_cfg_for_call is not None and getattr(angle_cfg_for_call, 'enable_speed_adaptive_sigma', False):
                        track_speeds = []
                        for trk in trackers:
                            try:
                                track_speeds.append(float(np.linalg.norm(get_velocity(trk)[:3])))
                            except Exception:
                                track_speeds.append(0.0)
                        angle_sigma = np.array([
                            compute_speed_adaptive_sigma(
                                speed=speed,
                                sigma_base=angle_cfg_for_call.angle_cost_sigma,
                                sigma_min=getattr(angle_cfg_for_call, 'speed_sigma_min', None),
                                sigma_max=getattr(angle_cfg_for_call, 'speed_sigma_max', None),
                                sigma_k=getattr(angle_cfg_for_call, 'speed_sigma_k', 0.25),
                            )
                            for speed in track_speeds
                        ], dtype=np.float32)
                        try:
                            angle_cfg_for_call.angle_cost_sigma = angle_sigma
                        except Exception:
                            pass
                    
                    adaptive_w = compute_adaptive_cost_matrix_weights(
                        track_angles, det_angles,
                        base_weights=base_w,
                        angle_weight_method='gaussian',
                        angle_sigma=angle_sigma,
                        verbose=False
                    )
                    # ========== 应用质量控制 ==========
                    # 质量矩阵: [n_dets, n_tracks]
                    # adaptive_w['angle']: [n_tracks, n_dets]
                    # 需要转置质量矩阵以匹配维度
                    quality_matrix_T = quality_matrix.T  # [n_tracks, n_dets]
                    
                    # 低质量检测 → 角度权重设为0（完全不使用角度）
                    # 高质量检测 → 使用完整的自适应角度权重
                    angle_weight_matrix = w_ang * adaptive_w['angle'] * quality_matrix_T  # shape: (trks, dets)
                    if use_heading_motion_consistency:
                        consistency_sim = _motion_heading_similarity_matrix(
                            track_angles,
                            det_angles,
                            trackers,
                            sigma=getattr(angle_cfg_for_call, 'heading_motion_sigma', 0.35),
                        )
                        angle_weight_matrix = w_ang * consistency_sim * quality_matrix_T
                    if ambiguity_triggered_angle and ambiguity_pair_mask_det_trk is not None:
                        ambiguity_pair_mask_trk_det = ambiguity_pair_mask_det_trk.T.astype(np.float32)
                        angle_weight_matrix = angle_weight_matrix * ambiguity_pair_mask_trk_det
                    
                    # ========== 调试：检查权重矩阵 ==========
                    if verbose_flag:
                        print(f"[权重调试] w_ang={w_ang:.3f}", flush=True)
                        print(f"[权重调试] adaptive_w['angle']: min={adaptive_w['angle'].min():.3f}, "
                              f"max={adaptive_w['angle'].max():.3f}, mean={adaptive_w['angle'].mean():.3f}", flush=True)
                        print(f"[权重调试] quality_matrix_T: min={quality_matrix_T.min():.3f}, "
                              f"max={quality_matrix_T.max():.3f}, mean={quality_matrix_T.mean():.3f}", flush=True)
                        print(f"[权重调试] angle_weight_matrix: min={angle_weight_matrix.min():.3f}, "
                              f"max={angle_weight_matrix.max():.3f}, mean={angle_weight_matrix.mean():.3f}", flush=True)


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
                    'angle': (
                        angle_weight_matrix
                        if (angle_weight_matrix is not None)
                        else (0.0 if ambiguity_triggered_angle else w_ang)
                    )
                },
                verbose=bool(getattr(angle_cfg_for_call, 'verbose', False))
            )

            # 赋值为 (dets x trks) 供 linear_assignment 使用
            # 应用距离门控：对超出距离阈值的对置以极大代价
            final_cost = fused_cost.T
            if dist_thr_vec is not None and dist_matrix is not None and dist_thr_vec.size == final_cost.shape[1]:
                gate = (dist_matrix <= dist_thr_vec.reshape(1, -1))
                # 对不满足门控的对设置极大代价
                final_cost[~gate] = 1e9
            matched_indices = linear_assignment(final_cost)
            if verbose_flag:
                try:
                    print(f"[L1 路径] fused_cost: matches={matched_indices.shape[0]}", flush=True)
                except Exception:
                    pass
    else:
        matched_indices = np.empty(shape=(0, 2))
        if verbose_flag:
            try:
                print(f"[L1 提示] 无法构建候选对: iou_matrix形状={iou_matrix.shape}", flush=True)
            except Exception:
                pass

    unmatched_detections = []
    for d, det in enumerate(dets_8corner):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trks_8corner):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    for m in matched_indices:
        dyn_iou_thr = iou_threshold
        if thr_vec is not None and m[1] < thr_vec.size:
            dyn_iou_thr = float(thr_vec[m[1]])
        iou_ok = (iou_matrix[m[0], m[1]] >= dyn_iou_thr)
        dist_ok = True
        dyn_dist_thr = None
        dist_val = None
        if dist_thr_vec is not None and dist_matrix is not None and m[1] < dist_thr_vec.size:
            dyn_dist_thr = float(dist_thr_vec[m[1]])
            dist_val = float(dist_matrix[m[0], m[1]])
            dist_ok = (dist_val <= dyn_dist_thr)
        # 动态门控总判定
        if (not iou_ok) or (not dist_ok):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            # 诊断日志: 若动态门控通过但静态门控失败，认为被RASSA拯救
            saved_msgs = []
            # 静态IoU门控为 iou_threshold
            if iou_threshold is not None:
                if iou_matrix[m[0], m[1]] < float(iou_threshold):
                    saved_msgs.append("IoU")
            # 静态距离门控为 base_dist
            if dist_val is not None and dyn_dist_thr is not None:
                if dist_val > float(base_dist):
                    saved_msgs.append("Dist")
            if len(saved_msgs) > 0:
                try:
                    print(f"[RASSA SAVED] det={m[0]} trk={m[1]} reason={'+'.join(saved_msgs)} iou={float(iou_matrix[m[0], m[1]]):.3f} dyn_iou={float(dyn_iou_thr):.3f} static_iou={float(iou_threshold):.3f} dist={(dist_val if dist_val is not None else float('nan')):.3f} dyn_dist={(dyn_dist_thr if dyn_dist_thr is not None else float('nan')):.3f} static_dist={float(base_dist):.3f}")
                except Exception:
                    pass
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
