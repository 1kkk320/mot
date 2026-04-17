# -*- coding: utf-8 -*-
"""
航向角质量评估模块

根据检测质量动态决定是否使用角度特征
低质量检测 → 不使用角度 → 回退到基础代价矩阵（外观+运动）
"""

import numpy as np
import math


def compute_angle_quality(detection, track=None):
    """
    计算单个检测的角度质量评分
    
    Args:
        detection: 检测对象
        track: 轨迹对象（可选，用于一致性检查）
    
    Returns:
        quality: 质量评分 [0, 1]
                 0 = 完全不可信（不使用角度）
                 1 = 完全可信（使用角度）
    """
    # 1. 基于检测置信度
    conf = getattr(detection, 'score', 1.0)
    
    # 置信度阈值：低于0.4直接返回0（不使用角度）
    if conf < 0.4:
        return 0.0
    
    # 置信度评分：归一化到[0, 1]
    # conf=0.4 → q_conf=0.0
    # conf=0.8 → q_conf=1.0
    q_conf = min(1.0, (conf - 0.4) / (0.8 - 0.4))
    
    # 2. 基于目标尺寸（大目标角度更准确）
    try:
        if hasattr(detection, 'bbox') and detection.bbox is not None and len(detection.bbox) >= 7:
            # bbox格式: [x, y, z, rot_y, l, w, h]
            length = float(detection.bbox[4])
            width = float(detection.bbox[5])
            area = length * width
            
            # 尺寸评分：
            # area < 5.0 → q_size=0.5（小目标，角度不太准）
            # area >= 10.0 → q_size=1.0（大目标，角度准确）
            if area < 5.0:
                q_size = 0.5
            elif area >= 10.0:
                q_size = 1.0
            else:
                q_size = 0.5 + (area - 5.0) / (10.0 - 5.0) * 0.5
        else:
            q_size = 0.7  # 默认中等
    except Exception:
        q_size = 0.7
    
    # 3. 基于历史一致性（如果有轨迹）
    if track is not None:
        try:
            from tracking.angle_feature import wrap_to_pi
            
            # 获取轨迹的平滑角度或原始角度
            track_angle = None
            if hasattr(track, 'angle_smoothed') and track.angle_smoothed is not None:
                track_angle = float(track.angle_smoothed)
            elif hasattr(track, 'pose') and track.pose is not None and len(track.pose) >= 7:
                track_angle = float(track.pose[3])
            
            # 获取检测角度
            det_angle = None
            if hasattr(detection, 'bbox') and detection.bbox is not None and len(detection.bbox) >= 7:
                det_angle = float(detection.bbox[3])
            
            if track_angle is not None and det_angle is not None:
                # 计算角度差
                delta = abs(wrap_to_pi(det_angle - track_angle))
                
                # 一致性评分：
                # delta < 10° → q_consistency=1.0（高度一致）
                # delta < 30° → q_consistency=0.6（中等一致）
                # delta >= 30° → q_consistency=0.3（不一致，可能是噪声）
                if delta < math.radians(10):
                    q_consistency = 1.0
                elif delta < math.radians(30):
                    q_consistency = 0.6
                else:
                    q_consistency = 0.3
                
                # 综合评分（有轨迹历史时）
                return 0.4 * q_conf + 0.3 * q_size + 0.3 * q_consistency
        
        except Exception:
            pass
    
    # 综合评分（无轨迹历史时）
    return 0.6 * q_conf + 0.4 * q_size


def compute_angle_quality_matrix(detections, tracks):
    """
    计算检测-轨迹对的角度质量矩阵
    
    Args:
        detections: 检测列表
        tracks: 轨迹列表
    
    Returns:
        quality_matrix: 质量矩阵 [n_dets, n_tracks]
                       每个元素表示该检测-轨迹对是否应该使用角度特征
                       0 = 不使用角度
                       1 = 使用角度
    """
    n_dets = len(detections)
    n_tracks = len(tracks)
    
    if n_dets == 0 or n_tracks == 0:
        return np.zeros((n_dets, n_tracks))
    
    quality_matrix = np.zeros((n_dets, n_tracks))
    
    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            quality = compute_angle_quality(det, trk)
            quality_matrix[d, t] = quality
    
    return quality_matrix


def should_use_angle(detection, track=None, threshold=0.5):
    """
    判断是否应该使用角度特征（二值决策）
    
    Args:
        detection: 检测对象
        track: 轨迹对象（可选）
        threshold: 质量阈值（默认0.5）
    
    Returns:
        bool: True=使用角度, False=不使用角度
    """
    quality = compute_angle_quality(detection, track)
    return quality >= threshold


def get_angle_usage_mask(detections, tracks, threshold=0.5):
    """
    获取角度使用掩码矩阵（二值）
    
    Args:
        detections: 检测列表
        tracks: 轨迹列表
        threshold: 质量阈值
    
    Returns:
        mask: 布尔矩阵 [n_dets, n_tracks]
              True = 使用角度
              False = 不使用角度（回退到基础代价）
    """
    quality_matrix = compute_angle_quality_matrix(detections, tracks)
    return quality_matrix >= threshold
