"""
代价矩阵融合模块 - 将多种特征融合为统一的关联代价

支持的特征:
1. IoU (位置重叠)
2. 速度相似度 (运动一致性)
3. 外观特征 (视觉相似度)
4. 航向角 (朝向一致性) - 新增

融合策略:
C = w_iou * c_iou + w_vel * c_vel + w_app * c_app + w_ang * c_ang

其中:
- c_iou: IoU代价 (1 - IoU)
- c_vel: 速度相似度代价 (1 - similarity)
- c_app: 外观代价 (1 - cosine_similarity)
- c_ang: 角度代价 (|Δθ|/π 或高斯惩罚)
"""

import numpy as np
import math
from tracking.angle_feature import (
    compute_angle_similarity_matrix,
    extract_angles_from_boxes,
    angle_gate
)


def normalize_cost_matrix(cost_matrix, method='minmax'):
    """
    将代价矩阵归一化到 [0, 1]
    
    Args:
        cost_matrix: 输入代价矩阵
        method: 'minmax' 或 'zscore'
        
    Returns:
        normalized_matrix: 归一化后的矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    if method == 'minmax':
        min_val = np.min(cost_matrix)
        max_val = np.max(cost_matrix)
        if max_val > min_val:
            return (cost_matrix - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(cost_matrix)
    elif method == 'zscore':
        mean_val = np.mean(cost_matrix)
        std_val = np.std(cost_matrix)
        if std_val > 0:
            normalized = (cost_matrix - mean_val) / std_val
            # 压缩到 [0, 1]
            return np.clip(normalized, -3, 3) / 6 + 0.5
        else:
            return np.ones_like(cost_matrix) * 0.5
    else:
        return cost_matrix


def fuse_cost_matrices(cost_dict, weights, normalize=True, verbose=False):
    """
    融合多个代价矩阵为单一代价矩阵
    
    Args:
        cost_dict: 代价字典 {'iou': matrix, 'velocity': matrix, 'appearance': matrix, 'angle': matrix}
        weights: 权重字典 {'iou': w_iou, 'velocity': w_vel, 'appearance': w_app, 'angle': w_ang}
        normalize: 是否先归一化各个代价矩阵
        verbose: 是否打印调试信息
        
    Returns:
        fused_cost: 融合后的代价矩阵 [n_tracks, n_dets]
    """
    # 获取矩阵形状
    shape = None
    for key, matrix in cost_dict.items():
        if matrix is not None and matrix.size > 0:
            shape = matrix.shape
            break
    
    if shape is None:
        return np.array([])
    
    # 初始化融合矩阵
    fused_cost = np.zeros(shape)
    total_weight = 0.0
    
    # 融合各个特征
    for feature_name, cost_matrix in cost_dict.items():
        if cost_matrix is None or cost_matrix.size == 0:
            continue
        
        weight = weights.get(feature_name, 0.0)
        if weight <= 0:
            continue
        
        # 归一化
        if normalize:
            normalized_cost = normalize_cost_matrix(cost_matrix, method='minmax')
        else:
            normalized_cost = cost_matrix
        
        # 加权融合
        fused_cost += weight * normalized_cost
        total_weight += weight
        
        if verbose:
            print(f"[融合] {feature_name}: weight={weight:.3f}, "
                  f"cost_range=[{normalized_cost.min():.3f}, {normalized_cost.max():.3f}]")
    
    # 归一化融合结果
    if total_weight > 0:
        fused_cost /= total_weight
    
    return fused_cost


def apply_angle_gate(cost_matrix, gate_mask):
    """
    应用角度门控: 被拒绝的候选项设为无穷大
    
    Args:
        cost_matrix: 代价矩阵 [n_tracks, n_dets]
        gate_mask: 门控掩码 [n_tracks, n_dets] (True 表示通过)
        
    Returns:
        gated_cost: 应用门控后的代价矩阵
    """
    gated_cost = cost_matrix.copy()
    gated_cost[~gate_mask] = np.inf  # 被拒绝的候选项设为无穷大
    return gated_cost


def compute_fused_cost_matrix(
    tracks,
    detections,
    iou_matrix=None,
    velocity_matrix=None,
    appearance_matrix=None,
    angle_config=None,
    weights=None,
    verbose=False
):
    """
    计算融合代价矩阵 (包含角度特征)
    
    Args:
        tracks: 轨迹列表
        detections: 检测列表
        iou_matrix: IoU代价矩阵 [n_tracks, n_dets] (可选)
        velocity_matrix: 速度相似度矩阵 [n_tracks, n_dets] (可选)
        appearance_matrix: 外观代价矩阵 [n_tracks, n_dets] (可选)
        angle_config: 角度配置对象 (AngleFeatureConfig)
        weights: 权重字典 (如果为None, 使用默认值)
        verbose: 是否打印调试信息
        
    Returns:
        fused_cost: 融合后的代价矩阵 [n_tracks, n_dets]
        angle_cost: 角度代价矩阵 (如果启用)
        gate_mask: 角度门控掩码 (如果启用)
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    
    if n_tracks == 0 or n_dets == 0:
        return np.array([]), None, None
    
    # 默认权重
    if weights is None:
        weights = {
            'iou': 0.4,
            'velocity': 0.3,
            'appearance': 0.15,
            'angle': 0.15
        }
    
    # 构建代价字典
    cost_dict = {}
    
    # IoU代价
    if iou_matrix is not None:
        cost_dict['iou'] = 1.0 - iou_matrix  # 转换为代价 (1 - IoU)
    
    # 速度代价
    if velocity_matrix is not None:
        cost_dict['velocity'] = 1.0 - velocity_matrix  # 转换为代价
    
    # 外观代价
    if appearance_matrix is not None:
        cost_dict['appearance'] = 1.0 - appearance_matrix  # 转换为代价
    
    # 角度代价 (新增)
    angle_cost = None
    gate_mask = None
    
    if angle_config is not None and angle_config.enable_angle_feature:
        try:
            # 提取轨迹和检测的角度
            track_angles = np.array([get_track_angle(t) for t in tracks])
            det_angles = np.array([get_detection_angle(d) for d in detections])
            
            if len(det_angles) == 0:
                det_angles = np.zeros(n_dets)
            
            # 计算角度代价矩阵
            angle_cost, gate_mask = compute_angle_similarity_matrix(
                track_angles,
                det_angles,
                method=angle_config.angle_cost_method,
                sigma=angle_config.angle_cost_sigma,
                gate_threshold=angle_config.angle_gate_threshold if angle_config.enable_angle_gate else None
            )
            
            cost_dict['angle'] = angle_cost
            
            if verbose:
                print(f"[角度特征] 启用, 方法={angle_config.angle_cost_method}, "
                      f"权重={weights.get('angle', 0.0):.3f}")
        
        except Exception as e:
            if verbose:
                print(f"[角度特征] 计算失败: {e}")
            angle_cost = None
            gate_mask = None
    
    # 融合代价矩阵
    fused_cost = fuse_cost_matrices(
        cost_dict,
        weights,
        normalize=True,
        verbose=verbose
    )
    
    # 应用角度门控
    if gate_mask is not None and angle_config.enable_angle_gate:
        fused_cost = apply_angle_gate(fused_cost, gate_mask)
        if verbose:
            n_rejected = (~gate_mask).sum()
            print(f"[角度门控] 拒绝 {n_rejected} 个候选项")
    
    return fused_cost, angle_cost, gate_mask


# ============ 辅助函数 ============

def get_track_angle(track):
    """从轨迹对象中提取角度 (优先3D yaw)"""
    # 优先 Track_3D.pose[6]
    if hasattr(track, 'pose') and isinstance(track.pose, (list, tuple, np.ndarray)) and len(track.pose) >= 7:
        return track.pose[6]
    # 其次 track.angle
    if hasattr(track, 'angle'):
        return track.angle
    # 再次 bbox[6] 或 bbox[4]
    if hasattr(track, 'bbox') and isinstance(track.bbox, (list, tuple, np.ndarray)):
        if len(track.bbox) >= 7:
            return track.bbox[6]
        if len(track.bbox) >= 5:
            return track.bbox[4]
    return 0.0


def get_detection_angle(detection):
    """从检测对象中提取角度 (优先3D yaw)"""
    if hasattr(detection, 'bbox') and isinstance(detection.bbox, (list, tuple, np.ndarray)):
        if len(detection.bbox) >= 7:
            return detection.bbox[6]
        if len(detection.bbox) >= 5:
            return detection.bbox[4]
    if hasattr(detection, 'angle'):
        return detection.angle
    if isinstance(detection, (list, tuple, np.ndarray)):
        if len(detection) >= 7:
            return detection[6]
        if len(detection) >= 5:
            return detection[4]
    return 0.0
