"""
虚拱速度历史补足模块 - 方案I
在二次关联成功时，根据缺失帧数线性补足velocity_history中的缺失帧
"""

import numpy as np
from tracking.cost_function import estimate_detection_velocity


def fill_velocity_history(track, detection, current_frame, prev_detections=None, verbose=False):
    """
    在二次关联成功后，补足缺失帧的速度历史
    
    原理:
    - Frame 100: 真实检测 → velocity_history = [(100, [10, 0, 0])]
    - Frame 101-102: 缺失 (未关联) → 不记录速度
    - Frame 103: 二次关联成功 → 补足缺失帧速度
    
    补足后:
    - velocity_history = [
        (100, [10, 0, 0]),           # 真实
        (101, [10.067, 0.033, 0]),   # 虚拱 (线性插值)
        (102, [10.133, 0.067, 0]),   # 虚拱 (线性插值)
        (103, [10.2, 0.1, 0])        # 真实
    ]
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        current_frame: 当前帧号
        prev_detections: 历史检测字典 (用于估计检测速度)
        verbose: 是否打印调试信息
    
    Returns:
        None (直接修改track.velocity_history)
    """
    
    frames_missed = track.time_since_update
    
    if frames_missed <= 1:
        # 缺失帧数少，不需要补足
        return
    
    # 获取最后一帧的速度和帧号
    if len(track.velocity_history) == 0:
        return
    
    last_frame_id, last_velocity = track.velocity_history[-1]
    
    # 估计当前检测的速度
    if prev_detections is not None:
        current_velocity = estimate_detection_velocity(detection, prev_detections, current_frame)
    else:
        # 如果没有历史检测，使用检测速度的简单估计
        current_velocity = np.zeros(3)
    
    # 线性插值补足缺失帧的速度
    virtual_velocities = []
    
    for i in range(1, frames_missed):
        # 进度比例 (0-1)
        progress = i / frames_missed
        
        # 线性插值速度
        # virtual_vel = last_vel + (current_vel - last_vel) * progress
        interpolated_velocity = last_velocity + (current_velocity - last_velocity) * progress
        
        # 虚拱帧号 (在缺失期间均匀分布)
        virtual_frame_id = last_frame_id + i
        
        # 添加到速度历史
        track.velocity_history.append((virtual_frame_id, interpolated_velocity.copy()))
        virtual_velocities.append((virtual_frame_id, interpolated_velocity.copy()))
        
        if verbose and i == 1:
            print(f"[虚拱速度补足] 轨迹{track.track_id_3d}:")
            print(f"  缺失帧数: {frames_missed}")
            print(f"  最后真实速度 (Frame {last_frame_id}): {last_velocity}")
            print(f"  当前检测速度: {current_velocity}")
            print(f"  虚拱速度 (Frame {virtual_frame_id}): {interpolated_velocity}")
    
    # 保持历史长度
    if len(track.velocity_history) > track.max_history_length:
        # 移除最早的记录
        removed_count = len(track.velocity_history) - track.max_history_length
        for _ in range(removed_count):
            track.velocity_history.pop(0)
    
    if verbose and len(virtual_velocities) > 0:
        print(f"  补足 {len(virtual_velocities)} 帧虚拱速度")


def fill_velocity_history_with_kf(track, detection, current_frame, prev_detections=None, verbose=False):
    """
    增强版本: 使用KF预测速度进行补足
    
    相比简单线性插值，使用KF预测速度可能更准确
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
        current_frame: 当前帧号
        prev_detections: 历史检测字典
        verbose: 是否打印调试信息
    
    Returns:
        None (直接修改track.velocity_history)
    """
    
    frames_missed = track.time_since_update
    
    if frames_missed <= 1:
        return
    
    if len(track.velocity_history) == 0:
        return
    
    last_frame_id, last_velocity = track.velocity_history[-1]
    
    # 估计当前检测的速度
    if prev_detections is not None:
        current_velocity = estimate_detection_velocity(detection, prev_detections, current_frame)
    else:
        current_velocity = np.zeros(3)
    
    # 获取预测速度趋势 (加速度)
    if hasattr(track, 'get_smooth_velocity_trend'):
        velocity_trend = track.get_smooth_velocity_trend(window=3)
    else:
        velocity_trend = np.zeros(3)
    
    # 线性插值补足缺失帧的速度
    for i in range(1, frames_missed):
        # 进度比例 (0-1)
        progress = i / frames_missed
        
        # 方法1: 简单线性插值
        interpolated_velocity = last_velocity + (current_velocity - last_velocity) * progress
        
        # 方法2: 考虑加速度的插值 (可选)
        # 如果速度变化很大，可能需要考虑加速度
        # interpolated_velocity = last_velocity + velocity_trend * i * 0.1 + (current_velocity - last_velocity) * progress
        
        # 虚拱帧号
        virtual_frame_id = last_frame_id + i
        
        # 添加到速度历史
        track.velocity_history.append((virtual_frame_id, interpolated_velocity.copy()))
    
    # 保持历史长度
    if len(track.velocity_history) > track.max_history_length:
        removed_count = len(track.velocity_history) - track.max_history_length
        for _ in range(removed_count):
            track.velocity_history.pop(0)
    
    if verbose:
        print(f"[虚拱速度补足-KF版] 轨迹{track.track_id_3d}:")
        print(f"  缺失帧数: {frames_missed}")
        print(f"  最后真实速度: {last_velocity}")
        print(f"  当前检测速度: {current_velocity}")
        print(f"  速度趋势: {velocity_trend}")
