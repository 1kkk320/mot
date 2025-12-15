"""
航向角特征模块 - 基于AB3DMOT思想的2D跟踪角度处理

核心功能:
1. wrap_to_pi: 将角度归一化到 [-π, π]
2. angle_cost: 计算角度代价 (归一化或高斯惩罚)
3. angle_gate: 角度门控 (粗筛不匹配的候选项)
4. compute_angle_similarity: 计算角度相似度矩阵
"""

import numpy as np
import math


def wrap_to_pi(angle):
    """
    将角度归一化到 [-π, π] 范围
    
    Args:
        angle: 输入角度 (弧度)
        
    Returns:
        wrapped_angle: 归一化后的角度 (在 [-π, π] 范围内)
    """
    # 处理 NaN 和无效值
    if np.isnan(angle) or angle is None:
        return 0.0
    
    # 使用 atan2 方法: 更稳定和高效
    # atan2(sin(x), cos(x)) 自动返回 [-π, π] 范围内的角度
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_cost_linear(angle_track, angle_det, sigma=None):
    """
    线性角度代价: c_angle = |Δθ| / π
    
    Args:
        angle_track: 轨迹角度 (弧度)
        angle_det: 检测角度 (弧度)
        sigma: 未使用 (保持接口一致)
        
    Returns:
        cost: [0, 1] 范围的代价值
    """
    if np.isnan(angle_track) or np.isnan(angle_det):
        return 1.0  # 无效角度视为完全不匹配
    
    delta_angle = wrap_to_pi(angle_track - angle_det)
    cost = abs(delta_angle) / math.pi
    return min(cost, 1.0)  # 确保在 [0, 1] 范围内


def angle_cost_gaussian(angle_track, angle_det, sigma=0.3):
    """
    高斯惩罚角度代价: c_angle = 1 - exp(-Δθ^2 / (2σ^2))
    
    Args:
        angle_track: 轨迹角度 (弧度)
        angle_det: 检测角度 (弧度)
        sigma: 高斯标准差 (默认0.3弧度 ≈ 17度)
        
    Returns:
        cost: [0, 1] 范围的代价值
    """
    if np.isnan(angle_track) or np.isnan(angle_det):
        return 1.0
    
    delta_angle = wrap_to_pi(angle_track - angle_det)
    # 高斯惩罚: 角度差越大, 代价越高
    cost = 1.0 - np.exp(-(delta_angle ** 2) / (2 * sigma ** 2))
    return min(cost, 1.0)


def angle_gate(angle_track, angle_det, threshold=math.pi/2):
    """
    角度门控: 检查角度差是否超过阈值
    
    用于粗筛: 如果 |Δθ| > threshold, 返回 False (不匹配)
    
    Args:
        angle_track: 轨迹角度 (弧度)
        angle_det: 检测角度 (弧度)
        threshold: 角度阈值 (默认 π/2 = 90度)
        
    Returns:
        pass_gate: True 表示通过门控 (可能匹配), False 表示被拒绝
    """
    if np.isnan(angle_track) or np.isnan(angle_det):
        return True  # 无效角度不做门控拒绝
    
    delta_angle = wrap_to_pi(angle_track - angle_det)
    return abs(delta_angle) <= threshold


def compute_angle_similarity_matrix(track_angles, det_angles, method='linear', sigma=0.3, gate_threshold=None):
    """
    计算角度相似度矩阵 (代价矩阵形式)
    
    Args:
        track_angles: 轨迹角度数组 [n_tracks]
        det_angles: 检测角度数组 [n_dets]
        method: 'linear' 或 'gaussian'
        sigma: 高斯方法的标准差
        gate_threshold: 门控阈值 (None 表示不做门控)
        
    Returns:
        angle_cost_matrix: [n_tracks, n_dets] 代价矩阵
        gate_mask: [n_tracks, n_dets] 布尔矩阵 (True 表示通过门控)
    """
    n_tracks = len(track_angles)
    n_dets = len(det_angles)
    
    angle_cost_matrix = np.zeros((n_tracks, n_dets))
    gate_mask = np.ones((n_tracks, n_dets), dtype=bool)
    
    # 选择代价函数
    if method == 'gaussian':
        cost_fn = lambda t, d: angle_cost_gaussian(t, d, sigma)
    else:
        cost_fn = lambda t, d: angle_cost_linear(t, d)
    
    # 计算矩阵
    for i, angle_t in enumerate(track_angles):
        for j, angle_d in enumerate(det_angles):
            angle_cost_matrix[i, j] = cost_fn(angle_t, angle_d)
            
            # 应用门控
            if gate_threshold is not None:
                gate_mask[i, j] = angle_gate(angle_t, angle_d, gate_threshold)
    
    return angle_cost_matrix, gate_mask


def extract_angles_from_boxes(boxes):
    """
    从检测框中提取角度信息
    
    假设框格式: [x1, y1, x2, y2, angle, ...] 或 [x, y, w, h, angle, ...]
    
    Args:
        boxes: 检测框数组 [n, >=5]
        
    Returns:
        angles: 角度数组 [n]
    """
    if boxes is None or len(boxes) == 0:
        return np.array([])
    
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    # 假设第5列 (索引4) 是角度
    if boxes.shape[1] >= 5:
        angles = boxes[:, 4]
    else:
        # 如果没有角度信息, 返回零角度
        angles = np.zeros(boxes.shape[0])
    
    return angles


def correct_track_orientation(track_angle, det_angle, correction_threshold=math.pi/2):
    """
    AB3DMOT 风格的轨迹角度校正
    
    当轨迹角度与检测角度差过大时 (> threshold), 对轨迹角度做 π 校正
    
    Args:
        track_angle: 轨迹当前角度 (弧度)
        det_angle: 检测角度 (弧度)
        correction_threshold: 校正触发阈值 (默认 π/2)
        
    Returns:
        corrected_angle: 校正后的轨迹角度
    """
    if np.isnan(track_angle) or np.isnan(det_angle):
        return track_angle
    
    delta_angle = wrap_to_pi(track_angle - det_angle)
    
    if abs(delta_angle) > correction_threshold:
        # 做 π 校正
        corrected_angle = wrap_to_pi(track_angle + math.pi)
        return corrected_angle
    
    return track_angle


# ============ 配置类 ============

class AngleFeatureConfig:
    """角度特征配置管理"""
    
    def __init__(self):
        # 启用/禁用
        self.enable_angle_feature = False
        
        # 代价函数方法
        self.angle_cost_method = 'linear'  # 'linear' 或 'gaussian'
        self.angle_cost_sigma = 0.3        # 高斯方法的标准差 (弧度)
        
        # 门控参数
        self.enable_angle_gate = True
        self.angle_gate_threshold = math.radians(35)  # 35度
        
        # 融合权重
        self.angle_weight = 0.15  # 角度权重 (不要设太高, AB3DMOT 建议 0.1-0.2)
        
        # 角度校正 (AB3DMOT 风格)
        self.enable_orientation_correction = False  # 默认关闭, 可选启用
        self.correction_threshold = math.pi / 2
        
        # 调试
        self.verbose = False
    
    def __repr__(self):
        return (
            f"AngleFeatureConfig(\n"
            f"  enable={self.enable_angle_feature},\n"
            f"  method={self.angle_cost_method},\n"
            f"  gate={self.enable_angle_gate} (threshold={self.angle_gate_threshold:.3f}),\n"
            f"  weight={self.angle_weight},\n"
            f"  correction={self.enable_orientation_correction}\n"
            f")"
        )
