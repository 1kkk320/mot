# Author: wangxy
# Emial: 1393196999@qq.com

import copy, math
import numpy as np
from numba import jit
from scipy.spatial import ConvexHull


def iou_batch(boxA, boxB):

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # 防止除零错误
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    
    iou = interArea / denominator

    return iou

@jit
def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@jit
def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s): outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0: return None
    return (outputList)


def iou3d(corners1, corners2):
    ''' Compute 3D bounding box IoU, only working for object parallel to ground

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    # inter_area = shapely_polygon_intersection(rect1, rect2)
    _, inter_area = convex_hull_intersection(rect1, rect2)

    # try:
    #   _, inter_area = convex_hull_intersection(rect1, rect2)
    # except ValueError:
    #   inter_area = 0
    # except scipy.spatial.qhull.QhullError:
    #   inter_area = 0

    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def eucliDistance(detection, track):
    # coefficient_det = math.sqrt(detection[0] ** 2 + detection[1] ** 2 + detection[2] ** 2)
    # coefficient_trk = math.sqrt(track[0] ** 2 + track[1] ** 2 + track[2] ** 2)
    # x = [i / coefficient_det for i in detection]
    # y = [k / coefficient_trk for k in track]
    # dist = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)
    dist = math.sqrt((detection[0] - track[0]) ** 2 + (detection[1] - track[1]) ** 2 + (detection[2] - track[2]) ** 2)
    # dist = 1 /(1+dist)   # Normalization
    return dist


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def convert_3dbox_to_8corner(bbox3d_input):
    ''' Takes an object's 3D box with the representation of [x,y,z,theta,l,w,h] and
        convert it to the 8 corners of the 3D box

        Returns:
            corners_3d: (8,3) array in in rect camera coord
    '''
    # compute rotational matrix around yaw axis
    bbox3d = copy.copy(bbox3d_input)

    R = roty(bbox3d[3])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners  这是什么东西
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack(
        [x_corners, y_corners, z_corners]))  # np.vstack([x_corners,y_corners,z_corners])   3*8按照竖直方向排列
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + bbox3d[0]  # x
    corners_3d[1, :] = corners_3d[1, :] + bbox3d[1]  # y
    corners_3d[2, :] = corners_3d[2, :] + bbox3d[2]  # z

    return np.transpose(corners_3d)


# ==================== 速度自适应回溯关联相关函数 ====================

def get_velocity(track):
    """
    从卡尔曼滤波器提取速度
    
    Args:
        track: Track_3D对象
    
    Returns:
        velocity: [dx, dy, dz] 速度向量 (m/s)
    """
    if hasattr(track, 'kf_3d') and hasattr(track.kf_3d, 'kf'):
        # 提取速度状态 (索引7,8,9对应dx,dy,dz)
        velocity = track.kf_3d.kf.x[7:10].flatten()
        return velocity
    else:
        return np.zeros(3)


def compute_velocity_similarity(track_vel, det_vel, alpha=1.0):
    """
    计算速度相似度 (余弦相似度)
    
    Args:
        track_vel: 轨迹速度 [dx, dy, dz]
        det_vel: 检测速度 (从历史估计) [dx, dy, dz]
        alpha: 速度权重系数 (保留用于未来扩展)
    
    Returns:
        similarity: 速度相似度 (0-1), 1表示完全相同
    """
    # 计算速度大小
    norm_track = np.linalg.norm(track_vel)
    norm_det = np.linalg.norm(det_vel)
    
    # 如果速度太小(接近静止),返回中等相似度
    if norm_track < 1e-6 or norm_det < 1e-6:
        return 0.5
    
    # 余弦相似度: cos(θ) = (a·b) / (|a||b|)
    cos_sim = np.dot(track_vel, det_vel) / (norm_track * norm_det)
    
    # 归一化到[0,1]: cos从[-1,1]映射到[0,1]
    similarity = (cos_sim + 1) / 2
    
    return similarity


def estimate_detection_velocity(detection, prev_detections, frame_id):
    """
    从历史检测估计当前检测的速度
    
    Args:
        detection: 当前检测对象
        prev_detections: 历史检测字典 {frame_id: [detections]}
        frame_id: 当前帧ID
    
    Returns:
        velocity: 估计的速度 [dx, dy, dz] (m/s)
    """
    # 如果没有历史帧,返回零速度
    if frame_id - 1 not in prev_detections:
        return np.zeros(3)
    
    # 获取上一帧的检测
    prev_frame_dets = prev_detections[frame_id - 1]
    
    # 寻找最近的历史检测 (基于位置距离)
    min_dist = float('inf')
    best_match = None
    
    for prev_det in prev_frame_dets:
        # 计算3D位置距离
        dist = np.linalg.norm(detection.bbox[:3] - prev_det.bbox[:3])
        if dist < min_dist:
            min_dist = dist
            best_match = prev_det
    
    # 如果找到匹配且距离合理(小于5米)
    if best_match is not None and min_dist < 5.0:
        # 计算速度: v = Δs / Δt
        # 假设KITTI数据集帧率为10Hz, dt=0.1s
        velocity = (detection.bbox[:3] - best_match.bbox[:3]) / 0.1
        return velocity
    else:
        # 没有找到合理的匹配,返回零速度
        return np.zeros(3)


# ========== 方案3: 多帧回溯 - 新增函数 ==========

def compute_velocity_trend_similarity(track, det_vel, use_smooth=True):
    """
    基于速度趋势的相似度 (方案3)
    考虑加速度，预测下一帧速度
    
    Args:
        track: 轨迹对象
        det_vel: 检测速度
        use_smooth: 是否使用平滑趋势 (降低噪声)
    
    Returns:
        similarity: 趋势相似度 [0, 1]
    """
    # 获取轨迹当前速度
    trk_vel = get_velocity(track)
    
    # 获取速度趋势 (加速度)
    if use_smooth and hasattr(track, 'get_smooth_velocity_trend'):
        # 使用平滑趋势 (降低噪声)
        trk_trend = track.get_smooth_velocity_trend(window=3)
    elif hasattr(track, 'get_velocity_trend'):
        # 使用简单趋势
        trk_trend = track.get_velocity_trend()
    else:
        # 如果没有趋势信息，退化为普通速度相似度
        return compute_velocity_similarity(trk_vel, det_vel)
    
    # 预测下一帧速度 (考虑加速度)
    # dt = 0.1s (假设10Hz帧率)
    predicted_vel = trk_vel + trk_trend * 0.1
    
    # 计算与检测速度的相似度
    similarity = compute_velocity_similarity(predicted_vel, det_vel)
    
    return similarity


def compute_smooth_velocity_similarity(track, det_vel, window=3):
    """
    基于平滑速度的相似度 (方案3)
    使用多帧平均速度，降低噪声影响
    
    Args:
        track: 轨迹对象
        det_vel: 检测速度
        window: 平滑窗口大小
    
    Returns:
        similarity: 相似度 [0, 1]
    """
    # 使用平均速度 (更稳定)
    if hasattr(track, 'get_average_velocity'):
        avg_vel = track.get_average_velocity(window)
    else:
        # 退化为当前速度
        avg_vel = get_velocity(track)
    
    # 计算相似度
    similarity = compute_velocity_similarity(avg_vel, det_vel)
    
    return similarity