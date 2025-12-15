"""
自适应航向角权重分配模块

核心思想:
- 当角度差异小时 (< 45°): 赋予高权重 (0.2-0.3)
- 当角度差异中等时 (45°-90°): 赋予中权重 (0.1-0.15)
- 当角度差异大时 (> 90°): 赋予低权重 (0.05-0.1) 或门控拒绝

这样可以在不完全拒绝的情况下，对不确定的匹配降低角度特征的影响。
"""

import numpy as np
import math


def compute_adaptive_angle_weight(angle_diff, method='sigmoid'):
    """
    根据角度差异计算自适应权重
    
    Args:
        angle_diff: 角度差异 (弧度, 在 [-π, π] 范围内)
        method: 权重计算方法
                - 'sigmoid': 使用sigmoid函数 (平滑过渡)
                - 'linear': 使用分段线性函数 (快速衰减)
                - 'gaussian': 使用高斯函数 (钟形分布)
    
    Returns:
        weight: 权重值 [0, 1]
    """
    # 确保角度差异在 [0, π] 范围内
    abs_diff = abs(angle_diff)
    if abs_diff > math.pi:
        abs_diff = 2 * math.pi - abs_diff
    
    if method == 'sigmoid':
        # Sigmoid方法: 在45°处权重为0.5
        # w(θ) = 1 / (1 + exp(k*(θ - θ_mid)))
        # θ_mid = π/4 (45°), k = 10 (陡峭度)
        theta_mid = math.pi / 4  # 45度
        k = 10  # 陡峭度参数
        weight = 1.0 / (1.0 + np.exp(k * (abs_diff - theta_mid)))
        return float(weight)
    
    elif method == 'linear':
        # 分段线性方法
        # θ ∈ [0, π/6]: w = 1.0 (30度以内完全信任)
        # θ ∈ [π/6, π/2]: w = 1 - (θ - π/6) / (π/3) (线性衰减)
        # θ ∈ [π/2, π]: w = 0.2 (90度以上低权重)
        
        if abs_diff <= math.pi / 6:  # 30度以内
            return 1.0
        elif abs_diff <= math.pi / 2:  # 30-90度
            # 线性从1.0衰减到0.2
            weight = 1.0 - (abs_diff - math.pi / 6) * (0.8 / (math.pi / 3))
            return max(0.2, weight)
        else:  # 90度以上
            return 0.2
    
    elif method == 'gaussian':
        # 高斯方法: 以0为中心的高斯分布
        # w(θ) = exp(-θ² / (2σ²))
        # σ = π/6 (30度标准差)
        sigma = math.pi / 6
        weight = np.exp(-(abs_diff ** 2) / (2 * sigma ** 2))
        return float(weight)
    
    else:
        raise ValueError(f"未知的权重计算方法: {method}")


def compute_adaptive_cost_matrix_weights(track_angles, det_angles, 
                                         base_weights=None, 
                                         angle_weight_method='sigmoid',
                                         verbose=False):
    """
    计算自适应的代价矩阵权重
    
    对于每一对 (轨迹, 检测), 根据角度差异调整角度特征的权重,
    同时调整其他特征的权重以保持总权重为1.0
    
    Args:
        track_angles: 轨迹角度数组 [n_tracks]
        det_angles: 检测角度数组 [n_dets]
        base_weights: 基础权重字典
                     {'iou': 0.4, 'velocity': 0.3, 'appearance': 0.15, 'angle': 0.15}
        angle_weight_method: 角度权重计算方法
        verbose: 是否打印调试信息
    
    Returns:
        adaptive_weights: 自适应权重矩阵字典
                         {
                           'iou': [n_tracks, n_dets],
                           'velocity': [n_tracks, n_dets],
                           'appearance': [n_tracks, n_dets],
                           'angle': [n_tracks, n_dets]
                         }
    """
    if base_weights is None:
        base_weights = {
            'iou': 0.4,
            'velocity': 0.3,
            'appearance': 0.15,
            'angle': 0.15
        }
    
    n_tracks = len(track_angles)
    n_dets = len(det_angles)
    
    # 初始化权重矩阵
    adaptive_weights = {
        'iou': np.zeros((n_tracks, n_dets)),
        'velocity': np.zeros((n_tracks, n_dets)),
        'appearance': np.zeros((n_tracks, n_dets)),
        'angle': np.zeros((n_tracks, n_dets))
    }
    
    # 计算基础权重 (不含角度)
    base_weight_sum = base_weights['iou'] + base_weights['velocity'] + base_weights['appearance']
    
    for t in range(n_tracks):
        for d in range(n_dets):
            # 计算角度差异
            angle_diff = track_angles[t] - det_angles[d]
            # 归一化到 [-π, π]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 计算自适应角度权重
            adaptive_angle_weight = compute_adaptive_angle_weight(
                angle_diff, 
                method=angle_weight_method
            )
            
            # 调整权重: 角度权重降低时, 其他权重相应提高
            # 策略: 保持其他权重的相对比例不变, 只调整角度权重
            adjusted_angle_weight = base_weights['angle'] * adaptive_angle_weight
            
            # 其他权重保持不变 (总权重可能不为1, 这在融合时会归一化)
            adaptive_weights['iou'][t, d] = base_weights['iou']
            adaptive_weights['velocity'][t, d] = base_weights['velocity']
            adaptive_weights['appearance'][t, d] = base_weights['appearance']
            adaptive_weights['angle'][t, d] = adjusted_angle_weight
            
            if verbose and t == 0 and d == 0:
                print(f"[自适应权重] 角度差异: {math.degrees(angle_diff):.1f}°, "
                      f"自适应系数: {adaptive_angle_weight:.3f}, "
                      f"调整后角度权重: {adjusted_angle_weight:.3f}")
    
    return adaptive_weights


def compute_adaptive_cost_matrix_normalized(track_angles, det_angles,
                                            base_weights=None,
                                            angle_weight_method='sigmoid',
                                            verbose=False):
    """
    计算自适应权重并保证总权重为1.0
    
    Args:
        track_angles: 轨迹角度数组
        det_angles: 检测角度数组
        base_weights: 基础权重
        angle_weight_method: 角度权重计算方法
        verbose: 是否打印调试信息
    
    Returns:
        normalized_weights: 归一化后的权重矩阵字典 (每行总权重为1.0)
    """
    adaptive_weights = compute_adaptive_cost_matrix_weights(
        track_angles, det_angles,
        base_weights=base_weights,
        angle_weight_method=angle_weight_method,
        verbose=verbose
    )
    
    n_tracks, n_dets = adaptive_weights['iou'].shape
    normalized_weights = {
        'iou': np.zeros((n_tracks, n_dets)),
        'velocity': np.zeros((n_tracks, n_dets)),
        'appearance': np.zeros((n_tracks, n_dets)),
        'angle': np.zeros((n_tracks, n_dets))
    }
    
    for t in range(n_tracks):
        for d in range(n_dets):
            # 计算总权重
            total_weight = (adaptive_weights['iou'][t, d] +
                          adaptive_weights['velocity'][t, d] +
                          adaptive_weights['appearance'][t, d] +
                          adaptive_weights['angle'][t, d])
            
            if total_weight > 0:
                # 归一化
                normalized_weights['iou'][t, d] = adaptive_weights['iou'][t, d] / total_weight
                normalized_weights['velocity'][t, d] = adaptive_weights['velocity'][t, d] / total_weight
                normalized_weights['appearance'][t, d] = adaptive_weights['appearance'][t, d] / total_weight
                normalized_weights['angle'][t, d] = adaptive_weights['angle'][t, d] / total_weight
    
    return normalized_weights


class AdaptiveAngleWeightConfig:
    """自适应角度权重配置"""
    
    def __init__(self):
        self.enable_adaptive_weight = True
        self.angle_weight_method = 'sigmoid'  # 'sigmoid', 'linear', 'gaussian'
        
        # 基础权重
        self.base_weights = {
            'iou': 0.4,
            'velocity': 0.3,
            'appearance': 0.15,
            'angle': 0.15
        }
        
        # Sigmoid参数
        self.sigmoid_theta_mid = math.pi / 4  # 45度
        self.sigmoid_k = 10  # 陡峭度
        
        # 高斯参数
        self.gaussian_sigma = math.pi / 6  # 30度标准差
        
        self.verbose = False
    
    def __repr__(self):
        return (f"AdaptiveAngleWeightConfig(\n"
                f"  enable={self.enable_adaptive_weight},\n"
                f"  method={self.angle_weight_method},\n"
                f"  base_weights={self.base_weights},\n"
                f"  verbose={self.verbose}\n"
                f")")
