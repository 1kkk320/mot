# -*-coding:utf-8-*
# author: wangxy
import numpy as np
from tracking.cost_function import iou3d, convert_3dbox_to_8corner, iou_batch, eucliDistance
import scipy.spatial as sp


def split_cosine_dist(dets, trks, affinity_thresh=0.55, pair_diff_thresh=0.6, hard_thresh=True):

    cos_dist = np.zeros((len(dets), len(trks)))

    for i in range(len(dets)):
        for j in range(len(trks)):

            cos_d = 1 - sp.distance.cdist(dets[i], trks[j], "cosine")  ## shape = 3x3
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
        trks_embs = np.asarray([trk_emb.emb.tolist() for trk_emb in tracks])

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
            emb_cost = None if (trks_embs.shape[0] == 0 or det_embs.shape[0] == 0) else trks_embs @ det_embs.T
        else:
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
def associate_detections_to_trackers_fusion(detections, trackers, aw_off, grid_off,mot_off, iou_threshold, det_embs=None, det_app = False):
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

    trks_embs = np.asarray([trk_emb.emb.tolist() for trk_emb in trackers])
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
        # 计算外观特征
        if grid_off:
            emb_cost = None if (trks_embs.shape[0] == 0 or det_embs.shape[0] == 0) else det_embs @ trks_embs.T
        else:
            emb_cost = split_cosine_dist(det_embs, trks_embs)
        w_assoc_emb = 0.75
        aw_param = 0.4

    matches = []
    if not det_app:
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a),axis=1)
            else:
                if not aw_off:
                    w_matrix = compute_aw_new_metric(emb_cost, w_assoc_emb, aw_param)
                    emb_cost *= w_matrix
                else:
                    emb_cost *= w_assoc_emb
                if not mot_off:
                    final_cost = -(iou_matrix+emb_cost)
                    matched_indices = linear_assignment(final_cost)
                else:
                    final_cost = -emb_cost
                    matched_indices = linear_assignment(final_cost)
        else:
            matched_indices = np.empty(shape=(0, 2))
    else:
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
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

