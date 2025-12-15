"""
虚拟轨迹更新模块 - 方案C: 速度平滑更新
用于在回溯关联成功后，校正卡尔曼滤波器的速度估计
"""

import numpy as np
from tracking.cost_function import get_velocity, estimate_detection_velocity


def smooth_velocity_update(track, detection, prev_detections, current_frame,
                          det_vel_weight=0.6, pred_vel_weight=0.4,
                          max_vel_change=2.0, verbose=False):
    """
    方案C: 速度平滑更新
    使用平滑速度和趋势校正卡尔曼滤波器
    
    在回溯关联成功后调用，校正速度估计误差
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典 {frame_id: [detections]}
        current_frame: 当前帧号
        det_vel_weight: 检测速度权重 (0.6)
        pred_vel_weight: 预测速度权重 (0.4)
        max_vel_change: 最大速度变化限制 (m/s)
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    # ========== 步骤1: 估计检测速度 ==========
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    # ========== 步骤2: 获取轨迹平滑速度 ==========
    if hasattr(track, 'get_average_velocity'):
        track_smooth_vel = track.get_average_velocity(window=3)
    else:
        track_smooth_vel = get_velocity(track)
    
    # ========== 步骤3: 获取速度趋势 (加速度) ==========
    if hasattr(track, 'get_smooth_velocity_trend'):
        track_trend = track.get_smooth_velocity_trend(window=3)
    elif hasattr(track, 'get_velocity_trend'):
        track_trend = track.get_velocity_trend()
    else:
        track_trend = np.zeros(3)
    
    # ========== 步骤4: 预测速度 (考虑加速度) ==========
    frames_missed = track.time_since_update
    predicted_vel = track_smooth_vel + track_trend * frames_missed * 0.1
    
    # ========== 步骤5: 融合预测速度和检测速度 ==========
    fused_vel = pred_vel_weight * predicted_vel + det_vel_weight * det_vel
    
    # ========== 步骤6: 速度变化限制 (防止过度校正) ==========
    current_vel = get_velocity(track)
    vel_change = fused_vel - current_vel
    vel_change_norm = np.linalg.norm(vel_change)
    
    if vel_change_norm > max_vel_change:
        # 限制速度变化
        vel_change = vel_change / vel_change_norm * max_vel_change
        fused_vel = current_vel + vel_change
    
    # ========== 步骤7: 虚拟更新卡尔曼滤波器速度 ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤8: 更新速度历史 ==========
    if hasattr(track, 'velocity_history'):
        track.velocity_history.append((track.age, fused_vel.copy()))
        if len(track.velocity_history) > track.max_history_length:
            track.velocity_history.pop(0)
    
    # ========== 步骤9: 执行正常更新 ==========
    track.update_3d(detection)
    
    # ========== 调试输出 ==========
    if verbose and frames_missed > 1:
        print(f"[虚拟更新] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
        print(f"  平滑速度: {track_smooth_vel}")
        print(f"  速度趋势: {track_trend}")
        print(f"  预测速度: {predicted_vel}")
        print(f"  检测速度: {det_vel}")
        print(f"  融合速度: {fused_vel}")
        print(f"  速度变化: {vel_change_norm:.3f} m/s")
    
    return track


def simple_velocity_update(track, detection, prev_detections, current_frame,
                          track_vel_weight=0.3, verbose=False):
    """
    方案A: 简单虚拟更新
    直接融合轨迹速度和检测速度
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典
        current_frame: 当前帧号
        track_vel_weight: 轨迹速度权重 (0.3)
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    # 估计检测速度
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    # 获取轨迹速度
    track_vel = get_velocity(track)
    
    # 融合速度 (检测速度权重更高)
    fused_vel = track_vel_weight * track_vel + (1 - track_vel_weight) * det_vel
    
    # 虚拟更新速度
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # 执行正常更新
    track.update_3d(detection)
    
    if verbose:
        print(f"[简单虚拟更新] 轨迹{track.track_id_3d}:")
        print(f"  轨迹速度: {track_vel}")
        print(f"  检测速度: {det_vel}")
        print(f"  融合速度: {fused_vel}")
    
    return track


def hybrid_velocity_update(track, detection, prev_detections, current_frame,
                          det_vel_weight=0.85, pred_vel_weight=0.15,
                          max_vel_change=2.0, verbose=False):
    """
    方案F: 混合速度更新 (推荐)
    虚拱设置融合速度，KF更新位置后恢复虚拱速度
    
    融合策略: fused_vel = 0.15 * predicted_vel + 0.85 * det_vel
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典
        current_frame: 当前帧号
        det_vel_weight: 检测速度权重 (0.85)
        pred_vel_weight: 预测速度权重 (0.15)
        max_vel_change: 最大速度变化限制 (m/s)
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    # ========== 步骤1: 估计检测速度 ==========
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    # ========== 步骤2: 获取轨迹平滑速度 ==========
    if hasattr(track, 'get_average_velocity'):
        track_smooth_vel = track.get_average_velocity(window=3)
    else:
        track_smooth_vel = get_velocity(track)
    
    # ========== 步骤3: 获取速度趋势 (加速度) ==========
    if hasattr(track, 'get_smooth_velocity_trend'):
        track_trend = track.get_smooth_velocity_trend(window=3)
    elif hasattr(track, 'get_velocity_trend'):
        track_trend = track.get_velocity_trend()
    else:
        track_trend = np.zeros(3)
    
    # ========== 步骤4: 预测速度 (考虑加速度) ==========
    frames_missed = track.time_since_update
    predicted_vel = track_smooth_vel + track_trend * frames_missed * 0.1
    
    # ========== 步骤5: 融合预测速度和检测速度 ==========
    fused_vel = pred_vel_weight * predicted_vel + det_vel_weight * det_vel
    
    # ========== 步骤6: 速度变化限制 (防止过度校正) ==========
    current_vel = get_velocity(track)
    vel_change = fused_vel - current_vel
    vel_change_norm = np.linalg.norm(vel_change)
    
    if vel_change_norm > max_vel_change:
        # 限制速度变化
        vel_change = vel_change / vel_change_norm * max_vel_change
        fused_vel = current_vel + vel_change
    
    # ========== 步骤7: 虚拱设置融合速度 (第一次覆盖) ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤8: 记录速度历史 ==========
    if hasattr(track, 'velocity_history'):
        track.velocity_history.append((track.age, fused_vel.copy()))
        if len(track.velocity_history) > track.max_history_length:
            track.velocity_history.pop(0)
    
    # ========== 步骤9a: 执行KF更新 (获得准确的位置) ==========
    track.kf_3d.update(detection.bbox)
    
    # ========== 步骤9b: 恢复虚拱设置的速度 (第二次覆盖) ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤10: 更新统计信息 ==========
    track.additional_info = detection.additional_info
    track.pose = np.concatenate(track.kf_3d.kf.x[:7], axis=0)
    track.hits += 1
    track.age += 1
    track.time_since_update = 0
    track.confidence = 1
    
    # ========== 步骤11: 更新轨迹状态 ==========
    if track.hits >= track.n_init:
        track.state = 2  # TrackState.Confirmed
    else:
        track.state = 1  # TrackState.Tentative
    
    if track.fusion_time_update >= 3:
        track.state = 4  # TrackState.Reactivate
    
    # ========== 调试输出 ==========
    if verbose and frames_missed > 1:
        print(f"[方案F混合更新] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
        print(f"  平滑速度: {track_smooth_vel}")
        print(f"  速度趋势: {track_trend}")
        print(f"  预测速度: {predicted_vel}")
        print(f"  检测速度: {det_vel}")
        print(f"  融合速度: {fused_vel}")
        print(f"  速度变化: {vel_change_norm:.3f} m/s")
    
    return track


def linear_velocity_update(track, detection, prev_detections, current_frame,
                          det_vel_weight=0.85, pred_vel_weight=0.15,
                          max_vel_change=2.0, verbose=False):
    """
    线性插值速度更新
    使用线性插值融合预测速度和检测速度
    
    融合策略: fused_vel = (predicted_vel + det_vel) / 2
    权重: 与hybrid相同 (0.15*pred + 0.85*det)
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典
        current_frame: 当前帧号
        max_vel_change: 最大速度变化限制 (m/s)
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    # ========== 步骤1: 估计检测速度 ==========
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    # ========== 步骤2: 获取轨迹平滑速度 ==========
    if hasattr(track, 'get_average_velocity'):
        track_smooth_vel = track.get_average_velocity(window=3)
    else:
        track_smooth_vel = get_velocity(track)
    
    # ========== 步骤3: 获取速度趋势 (加速度) ==========
    if hasattr(track, 'get_smooth_velocity_trend'):
        track_trend = track.get_smooth_velocity_trend(window=3)
    elif hasattr(track, 'get_velocity_trend'):
        track_trend = track.get_velocity_trend()
    else:
        track_trend = np.zeros(3)
    
    # ========== 步骤4: 预测速度 (考虑加速度) ==========
    frames_missed = track.time_since_update
    predicted_vel = track_smooth_vel + track_trend * frames_missed * 0.1
    
    # ========== 步骤5: 线性插值融合 (使用与hybrid相同的权重) ==========
    fused_vel = pred_vel_weight * predicted_vel + det_vel_weight * det_vel
    
    # ========== 步骤6: 速度变化限制 ==========
    current_vel = get_velocity(track)
    vel_change = fused_vel - current_vel
    vel_change_norm = np.linalg.norm(vel_change)
    
    if vel_change_norm > max_vel_change:
        vel_change = vel_change / vel_change_norm * max_vel_change
        fused_vel = current_vel + vel_change
    
    # ========== 步骤7: 虚拱设置融合速度 ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤8: 记录速度历史 ==========
    if hasattr(track, 'velocity_history'):
        track.velocity_history.append((track.age, fused_vel.copy()))
        if len(track.velocity_history) > track.max_history_length:
            track.velocity_history.pop(0)
    
    # ========== 步骤9a: 执行KF更新 ==========
    track.kf_3d.update(detection.bbox)
    
    # ========== 步骤9b: 恢复虚拱设置的速度 ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤10: 更新统计信息 ==========
    track.additional_info = detection.additional_info
    track.pose = np.concatenate(track.kf_3d.kf.x[:7], axis=0)
    track.hits += 1
    track.age += 1
    track.time_since_update = 0
    track.confidence = 1
    
    # ========== 步骤11: 更新轨迹状态 ==========
    if track.hits >= track.n_init:
        track.state = 2  # TrackState.Confirmed
    else:
        track.state = 1  # TrackState.Tentative
    
    if track.fusion_time_update >= 3:
        track.state = 4  # TrackState.Reactivate
    
    # ========== 调试输出 ==========
    if verbose and frames_missed > 1:
        print(f"[线性插值更新] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
        print(f"  预测速度: {predicted_vel}")
        print(f"  检测速度: {det_vel}")
        print(f"  融合速度: {fused_vel}")
    
    return track


def exponential_velocity_update(track, detection, prev_detections, current_frame,
                               det_vel_weight=0.85, pred_vel_weight=0.15,
                               max_vel_change=2.0, verbose=False):
    """
    指数加权速度更新
    使用指数加权融合预测速度和检测速度，更信任检测速度
    
    融合策略: fused_vel = 0.15 * predicted_vel + 0.85 * det_vel
    权重: 与hybrid相同 (0.15*pred + 0.85*det)
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典
        current_frame: 当前帧号
        max_vel_change: 最大速度变化限制 (m/s)
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    # ========== 步骤1: 估计检测速度 ==========
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    # ========== 步骤2: 获取轨迹平滑速度 ==========
    if hasattr(track, 'get_average_velocity'):
        track_smooth_vel = track.get_average_velocity(window=3)
    else:
        track_smooth_vel = get_velocity(track)
    
    # ========== 步骤3: 获取速度趋势 (加速度) ==========
    if hasattr(track, 'get_smooth_velocity_trend'):
        track_trend = track.get_smooth_velocity_trend(window=3)
    elif hasattr(track, 'get_velocity_trend'):
        track_trend = track.get_velocity_trend()
    else:
        track_trend = np.zeros(3)
    
    # ========== 步骤4: 预测速度 (考虑加速度) ==========
    frames_missed = track.time_since_update
    predicted_vel = track_smooth_vel + track_trend * frames_missed * 0.1
    
    # ========== 步骤5: 指数加权融合 (使用与hybrid相同的权重) ==========
    fused_vel = pred_vel_weight * predicted_vel + det_vel_weight * det_vel
    
    # ========== 步骤6: 速度变化限制 ==========
    current_vel = get_velocity(track)
    vel_change = fused_vel - current_vel
    vel_change_norm = np.linalg.norm(vel_change)
    
    if vel_change_norm > max_vel_change:
        vel_change = vel_change / vel_change_norm * max_vel_change
        fused_vel = current_vel + vel_change
    
    # ========== 步骤7: 虚拱设置融合速度 ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤8: 记录速度历史 ==========
    if hasattr(track, 'velocity_history'):
        track.velocity_history.append((track.age, fused_vel.copy()))
        if len(track.velocity_history) > track.max_history_length:
            track.velocity_history.pop(0)
    
    # ========== 步骤9a: 执行KF更新 ==========
    track.kf_3d.update(detection.bbox)
    
    # ========== 步骤9b: 恢复虚拱设置的速度 ==========
    track.kf_3d.kf.x[7:10] = fused_vel.reshape((3, 1))
    
    # ========== 步骤10: 更新统计信息 ==========
    track.additional_info = detection.additional_info
    track.pose = np.concatenate(track.kf_3d.kf.x[:7], axis=0)
    track.hits += 1
    track.age += 1
    track.time_since_update = 0
    track.confidence = 1
    
    # ========== 步骤11: 更新轨迹状态 ==========
    if track.hits >= track.n_init:
        track.state = 2  # TrackState.Confirmed
    else:
        track.state = 1  # TrackState.Tentative
    
    if track.fusion_time_update >= 3:
        track.state = 4  # TrackState.Reactivate
    
    # ========== 调试输出 ==========
    if verbose and frames_missed > 1:
        print(f"[指数加权更新] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
        print(f"  预测速度: {predicted_vel}")
        print(f"  检测速度: {det_vel}")
        print(f"  融合速度: {fused_vel}")
    
    return track


def virtual_intermediate_update(track, detection, prev_detections, current_frame,
                               verbose=False):
    """
    方案B: 虚拟中间帧更新
    模拟缺失的中间帧
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        prev_detections: 历史检测字典
        current_frame: 当前帧号
        verbose: 是否打印调试信息
    
    Returns:
        track: 更新后的轨迹
    """
    
    frames_missed = track.time_since_update
    
    if frames_missed <= 1:
        # 缺失帧数少，直接更新
        track.update_3d(detection)
        return track
    
    # 估计检测速度
    det_vel = estimate_detection_velocity(detection, prev_detections, current_frame)
    
    if verbose:
        print(f"[虚拟中间帧更新] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
    
    # 虚拟更新中间帧
    for i in range(frames_missed - 1):
        # 虚拟位置 (线性插值)
        progress = (i + 1) / frames_missed
        virtual_pos = track.pose[:3] + det_vel[:3] * 0.1 * progress
        
        # 虚拟bbox (保持尺寸和角度)
        virtual_bbox = np.concatenate([
            virtual_pos,
            detection.bbox[3:]  # 尺寸和角度保持不变
        ])
        
        # 虚拟更新卡尔曼滤波器 (不改变hits)
        track.kf_3d.update(virtual_bbox)
        
        if verbose and i == 0:
            print(f"  虚拟更新位置: {virtual_pos}")
    
    # 正常更新
    track.update_3d(detection)
    
    return track


class VirtualUpdateConfig:
    """虚拟更新配置管理类"""
    
    def __init__(self):
        # 启用开关
        self.enable_virtual_update = True
        
        # 触发条件
        self.virtual_update_threshold = 1  # 中断帧数阈值
        
        # 方案选择
        self.update_method = 'smooth'  # 'simple', 'smooth', 'intermediate'
        
        # 参数配置
        self.det_vel_weight = 0.6      # 检测速度权重
        self.pred_vel_weight = 0.4     # 预测速度权重
        self.max_vel_change = 2.0      # 最大速度变化 (m/s)
        
        # 调试
        self.verbose = False
    
    def should_virtual_update(self, track):
        """判断是否需要虚拟更新"""
        if not self.enable_virtual_update:
            return False
        
        if track.time_since_update <= self.virtual_update_threshold:
            return False
        
        return True
    
    def apply_virtual_update(self, track, detection, prev_detections, current_frame):
        """应用虚拟更新"""
        
        if self.update_method == 'simple':
            return simple_velocity_update(
                track, detection, prev_detections, current_frame,
                track_vel_weight=1 - self.det_vel_weight,
                verbose=self.verbose
            )
        
        elif self.update_method == 'smooth':
            return smooth_velocity_update(
                track, detection, prev_detections, current_frame,
                det_vel_weight=self.det_vel_weight,
                pred_vel_weight=self.pred_vel_weight,
                max_vel_change=self.max_vel_change,
                verbose=self.verbose
            )
        
        elif self.update_method == 'hybrid':
            return hybrid_velocity_update(
                track, detection, prev_detections, current_frame,
                det_vel_weight=self.det_vel_weight,
                pred_vel_weight=self.pred_vel_weight,
                max_vel_change=self.max_vel_change,
                verbose=self.verbose
            )
        
        elif self.update_method == 'linear':
            return linear_velocity_update(
                track, detection, prev_detections, current_frame,
                det_vel_weight=self.det_vel_weight,
                pred_vel_weight=self.pred_vel_weight,
                max_vel_change=self.max_vel_change,
                verbose=self.verbose
            )
        
        elif self.update_method == 'exponential':
            return exponential_velocity_update(
                track, detection, prev_detections, current_frame,
                det_vel_weight=self.det_vel_weight,
                pred_vel_weight=self.pred_vel_weight,
                max_vel_change=self.max_vel_change,
                verbose=self.verbose
            )
        
        elif self.update_method == 'intermediate':
            return virtual_intermediate_update(
                track, detection, prev_detections, current_frame,
                verbose=self.verbose
            )
        
        else:
            # 默认方案
            return track.update_3d(detection)
