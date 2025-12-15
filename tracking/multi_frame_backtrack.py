"""
多帧关联模块 - 使用衰减时间因子
用于恢复超过3帧未关联的轨迹
"""

import numpy as np
from tracking.cost_function import get_velocity


class MultiFrameBacktrackConfig:
    """多帧回溯配置管理类"""
    
    def __init__(self):
        # 启用开关
        self.enable_multi_frame_backtrack = True
        
        # 触发条件
        self.min_backtrack_age = 3              # 最少缺失3帧
        self.max_backtrack_age = 15             # 最多缺失15帧
        
        # 衰减系数
        self.lambda_decay = 0.1                 # 衰减系数 (0.05-0.2)
        
        # 代价阈值
        self.cost_threshold = -0.5              # 代价阈值 (-1.0 ~ 0)
        
        # 检测缓冲
        self.detection_buffer_size = 30         # 保留30帧检测
        
        # 权重配置
        self.iou_weight = 0.5
        self.velocity_weight = 0.3
        self.appearance_weight = 0.2
        
        # 调试
        self.verbose = False


def compute_decay_factor(time_diff, lambda_decay=0.1):
    """
    计算衰减因子
    
    Args:
        time_diff: 时间差 (帧数)
        lambda_decay: 衰减系数
    
    Returns:
        decay: 衰减因子 (0-1]
    """
    decay = np.exp(-lambda_decay * time_diff)
    return decay


def compute_iou_3d(pose1, bbox2):
    """
    计算3D IoU (简化版，基于位置和尺寸)
    
    Args:
        pose1: 轨迹pose [x, y, z, theta, l, w, h]
        bbox2: 检测bbox [x, y, z, theta, l, w, h]
    
    Returns:
        iou: IoU值 (0-1)
    """
    # 提取位置和尺寸
    pos1 = pose1[:3]
    size1 = pose1[4:7]
    
    pos2 = bbox2[:3]
    size2 = bbox2[4:7]
    
    # 计算位置距离
    pos_dist = np.linalg.norm(pos1 - pos2)
    
    # 计算尺寸相似度
    size_sim = np.minimum(size1, size2).sum() / np.maximum(size1, size2).sum()
    
    # 简化IoU: 结合位置和尺寸
    # 距离越近，IoU越高
    iou = size_sim * np.exp(-pos_dist / 2.0)
    
    return np.clip(iou, 0, 1)


def compute_velocity_similarity(track, detection):
    """
    计算速度相似度
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
    
    Returns:
        similarity: 相似度 (0-1)
    """
    track_vel = get_velocity(track)
    
    # 估计检测速度 (如果有bbox信息)
    if hasattr(detection, 'velocity'):
        det_vel = detection.velocity
    else:
        det_vel = np.zeros(3)
    
    # 余弦相似度
    track_vel_norm = np.linalg.norm(track_vel)
    det_vel_norm = np.linalg.norm(det_vel)
    
    if track_vel_norm < 1e-6 or det_vel_norm < 1e-6:
        return 0.5  # 静止物体，给予中等相似度
    
    cos_sim = np.dot(track_vel, det_vel) / (track_vel_norm * det_vel_norm + 1e-6)
    similarity = (cos_sim + 1) / 2  # 转换到 [0, 1]
    
    return np.clip(similarity, 0, 1)


def compute_appearance_similarity(track, detection):
    """
    计算外观相似度
    
    Args:
        track: Track_3D对象
        detection: Detection_3D对象
    
    Returns:
        similarity: 相似度 (0-1)
    """
    # 简化版: 如果有嵌入向量，计算余弦相似度
    if hasattr(track, 'emb') and hasattr(detection, 'feature') and \
       track.emb is not None and detection.feature is not None:
        
        track_emb = track.emb / (np.linalg.norm(track.emb) + 1e-6)
        det_emb = detection.feature / (np.linalg.norm(detection.feature) + 1e-6)
        
        similarity = np.dot(track_emb, det_emb)
        return np.clip(similarity, 0, 1)
    
    return 0.5  # 默认中等相似度


def compute_decay_cost_matrix(track, detection_buffer, current_frame, 
                             config=None):
    """
    计算衰减代价矩阵
    
    Args:
        track: Track_3D对象
        detection_buffer: 检测缓冲 {frame_id: [detections]}
        current_frame: 当前帧号
        config: MultiFrameBacktrackConfig对象
    
    Returns:
        candidates: [(frame_id, detection, cost), ...] 按代价排序
    """
    if config is None:
        config = MultiFrameBacktrackConfig()
    
    candidates = []
    
    for frame_id, detections in detection_buffer.items():
        # 计算时间差
        time_diff = current_frame - frame_id
        
        # 跳过当前帧和未来帧
        if time_diff <= 0:
            continue
        
        # 计算衰减因子
        decay = compute_decay_factor(time_diff, config.lambda_decay)
        
        for det in detections:
            # 计算各项相似度
            iou = compute_iou_3d(track.pose, det.bbox)
            vel_sim = compute_velocity_similarity(track, det)
            app_sim = compute_appearance_similarity(track, det)
            
            # 融合相似度 (权重和为1)
            combined_sim = (config.iou_weight * iou + 
                           config.velocity_weight * vel_sim + 
                           config.appearance_weight * app_sim)
            
            # 应用衰减因子
            decayed_sim = combined_sim * decay
            
            # 转换为代价 (负数，用于匈牙利算法)
            cost = -decayed_sim
            
            candidates.append((frame_id, det, cost, time_diff, decay))
    
    # 按代价排序 (最小代价优先)
    candidates.sort(key=lambda x: x[2])
    
    return candidates


def multi_frame_backtrack_association(unmatched_tracks, detection_buffer, 
                                     current_frame, config=None):
    """
    执行多帧回溯关联
    
    Args:
        unmatched_tracks: 未匹配的轨迹列表
        detection_buffer: 检测缓冲 {frame_id: [detections]}
        current_frame: 当前帧号
        config: MultiFrameBacktrackConfig对象
    
    Returns:
        matched_pairs: [(track, detection, detection_frame_id), ...]
    """
    if config is None:
        config = MultiFrameBacktrackConfig()
    
    if not config.enable_multi_frame_backtrack:
        return []
    
    matched_pairs = []
    used_detections = set()
    
    for track in unmatched_tracks:
        # 检查是否满足触发条件
        if not (config.min_backtrack_age <= track.time_since_update <= config.max_backtrack_age):
            continue
        
        # 获取衰减代价矩阵
        candidates = compute_decay_cost_matrix(
            track, detection_buffer, current_frame, config
        )
        
        if not candidates:
            continue
        
        # 获取最佳候选
        best_frame_id, best_det, best_cost, time_diff, decay = candidates[0]
        
        # 检查代价阈值
        if best_cost < config.cost_threshold:
            # 检查检测是否已被使用
            det_id = id(best_det)
            if det_id not in used_detections:
                matched_pairs.append((track, best_det, best_frame_id, time_diff, decay))
                used_detections.add(det_id)
                
                if config.verbose:
                    print(f"[多帧关联] 轨迹{track.track_id_3d}:")
                    print(f"  缺失帧数: {track.time_since_update}")
                    print(f"  匹配检测帧: {best_frame_id} (时间差: {time_diff})")
                    print(f"  衰减因子: {decay:.4f}")
                    print(f"  代价: {best_cost:.4f}")
    
    return matched_pairs


def process_multi_frame_matches(matched_pairs, virtual_update_config=None, 
                               current_frame=None, verbose=False):
    """
    处理多帧匹配结果
    
    Args:
        matched_pairs: [(track, detection, detection_frame_id, time_diff, decay), ...]
        virtual_update_config: 虚拱更新配置
        current_frame: 当前帧号
        verbose: 是否打印调试信息
    
    Returns:
        updated_tracks: 更新后的轨迹列表
    """
    updated_tracks = []
    
    for track, detection, detection_frame_id, time_diff, decay in matched_pairs:
        # 计算预测速度 (使用加速度模型)
        if hasattr(track, 'get_average_velocity') and hasattr(track, 'get_smooth_velocity_trend'):
            smooth_vel = track.get_average_velocity(window=3)
            trend = track.get_smooth_velocity_trend(window=3)
            
            # 预测速度: 考虑时间差和加速度
            predicted_vel = smooth_vel + trend * time_diff * 0.1
        else:
            predicted_vel = get_velocity(track)
        
        # 虚拱更新速度
        track.kf_3d.kf.x[7:10] = predicted_vel.reshape((3, 1))
        
        # 执行更新
        track.update_3d(detection)
        
        # 标记为多帧回溯
        track.fusion_time_update += 1
        
        # 记录衰减因子 (用于调试)
        if not hasattr(track, 'last_decay_factor'):
            track.last_decay_factor = decay
        else:
            track.last_decay_factor = decay
        
        updated_tracks.append(track)
        
        if verbose:
            print(f"[多帧更新完成] 轨迹{track.track_id_3d}:")
            print(f"  预测速度: {predicted_vel}")
            print(f"  加速度: {trend}")
            print(f"  衰减因子: {decay:.4f}")
    
    return updated_tracks
