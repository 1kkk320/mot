"""
多帧关联模块 - 使用衰减时间因子
用于恢复超过3帧未关联的轨迹
"""

import os
import math
import numpy as np
from tracking.cost_function import get_velocity, compute_adaptive_weight_linear, estimate_detection_velocity, compute_velocity_similarity as compute_velocity_similarity_vec
from tracking.matching import linear_assignment, compute_rotated_ground_similarity
from tracking.motion_reliability import MotionReliabilityCalibrator


class MultiFrameBacktrackConfig:
    """多帧回溯配置管理类"""
    
    def __init__(self):
        # 启用开关
        self.enable_multi_frame_backtrack = True
        
        # 触发条件
        self.min_backtrack_age = 4              # 最少缺失3帧
        self.max_backtrack_age = 15             # 最多缺失15帧
        
        # 衰减系数
        self.lambda_decay = 0.12                # 衰减系数 (0.05-0.2)
        
        # 代价阈值
        self.cost_threshold = -0.55             # 代价阈值 (-1.0 ~ 0)
        
        # 检测缓冲
        self.detection_buffer_size = 30         # 保留30帧检测
        
        # 权重配置
        self.iou_weight = 0.5
        self.velocity_weight = 0.3
        self.appearance_weight = 0.2
        self.appearance_hard_gate = 0.6
        
        # 调试
        self.verbose = False
        # 最近帧窗口与候选控制
        self.last_k_frames = 5
        self.topk_per_frame = 1
        # 速度自适应参数
        self.vmax_for_adaptive_weight = 12.0
        # 协方差不确定性归一化尺度（m）
        self.uncertainty_norm = 12.0
        self.velocity_confidence_floor = 0.25
        self.enable_confidence_aware_motion_l25 = True
        self.l25_motion_reliability_mode = 'manual'
        self.l25_motion_reliability_model_path = ''
        self.l25_motion_reliability_feature_names = list(MotionReliabilityCalibrator.DEFAULT_FEATURES)
        self.l25_motion_reliability_bias = -0.1
        self.l25_motion_reliability_score_gain = 2.0
        self.l25_motion_reliability_uncertainty_gain = 2.0
        self.l25_velocity_share_min = 0.05
        self.l25_velocity_share_max = 0.30
        self.l25_velocity_geom_center = 0.30
        self.l25_velocity_geom_gain = 10.0
        self.l25_velocity_motion_center = 0.65
        self.l25_velocity_motion_gain = 8.0
        self.l25_velocity_reliability_strength = 0.60
        # 使用全局最优（线性分配）代替贪心
        self.use_global_assignment = True
        # 非线性回推开关（考虑加速度）
        self.use_nonlinear_backtrack = False    # 基础开关
        # 加速度门控：只在加速度显著时使用非线性
        self.use_acceleration_gate = True       # ✅ 启用加速度门控
        self.acceleration_threshold = 1.5       # 加速度阈值 (m/s²)，超过此值才用非线性
        self.enable_covariance_diag_log = False
        self.covariance_diag_log_path = None
        self.current_seq_id = None
        self.current_data_frame = None
        self.enable_hit_event_log = False
        self.hit_event_log_path = None
        self.enable_final_hit_event_log = False
        self.final_hit_event_log_path = None
        self.enable_l25_cooldown = False
        self.l25_cooldown_frames = 8
        self.allowed_backtrack_dts = None
        self.enable_candidate_pre_gate = True
        self.candidate_min_iou = 0.03
        self.candidate_min_size_ratio = 0.55
        self.candidate_max_center_dist_base = 2.0
        self.candidate_max_center_dist_per_dt = 0.35
        self.geometry_mode = 'box_iou'
        self.use_rotated_geom_in_l25 = False
        self.rotated_geom_weight_l25 = 0.10
        self.use_l25_memory_bank_appearance = False
        self.memory_bank_size = 3
        self.memory_bank_min_conf = 0.4
        self.memory_bank_rescore_margin = 0.03
        self.enable_memory_bank_stats_log = False
        self.memory_bank_stats_log_path = None
        self.memory_bank_stats = None
        self.enable_candidate_diag_log = False
        self.candidate_diag_log_path = None
        self._l25_motion_reliability_calibrator = None
        self._l25_motion_reliability_signature = None


def reset_memory_bank_stats(config):
    if config is None:
        return
    config.memory_bank_stats = {
        'calls': 0,
        'pairs_considered': 0,
        'pairs_with_memory_bank': 0,
        'pairs_app_changed': 0,
        'pairs_cost_changed': 0,
        'assignment_calls': 0,
        'assignment_changed_calls': 0,
        'assignment_changed_pairs': 0,
        'ambiguous_rows': 0,
        'ambiguous_cols': 0,
        'rescored_pairs': 0,
    }


def _ensure_memory_bank_stats(config):
    if config is None:
        return None
    if getattr(config, 'memory_bank_stats', None) is None:
        reset_memory_bank_stats(config)
    return config.memory_bank_stats


def append_memory_bank_stats_log(config):
    if config is None or not getattr(config, 'enable_memory_bank_stats_log', False):
        return
    log_path = getattr(config, 'memory_bank_stats_log_path', None)
    stats = _ensure_memory_bank_stats(config)
    if not log_path or stats is None:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        seq_id = getattr(config, 'current_seq_id', 'unknown')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                '[L2.5 MemoryBank Stats][{}] calls={} pairs_considered={} pairs_with_memory_bank={} '
                'pairs_app_changed={} pairs_cost_changed={} assignment_calls={} '
                'assignment_changed_calls={} assignment_changed_pairs={} ambiguous_rows={} ambiguous_cols={} rescored_pairs={}\n'.format(
                    seq_id,
                    int(stats.get('calls', 0)),
                    int(stats.get('pairs_considered', 0)),
                    int(stats.get('pairs_with_memory_bank', 0)),
                    int(stats.get('pairs_app_changed', 0)),
                    int(stats.get('pairs_cost_changed', 0)),
                    int(stats.get('assignment_calls', 0)),
                    int(stats.get('assignment_changed_calls', 0)),
                    int(stats.get('assignment_changed_pairs', 0)),
                    int(stats.get('ambiguous_rows', 0)),
                    int(stats.get('ambiguous_cols', 0)),
                    int(stats.get('rescored_pairs', 0)),
                )
            )
    except Exception:
        pass


def _append_l25_candidate_diag_log(
    config,
    track,
    detection,
    detection_frame_id,
    dt,
    stage,
    decision,
    reason,
    iou=None,
    app_sim=None,
    vel_sim=None,
    decay=None,
    uncertainty=None,
    motion_reliability=None,
    w_iou=None,
    w_vel=None,
    w_app=None,
    vel_share=None,
    vel_focus=None,
    combined_sim=None,
    cost=None,
):
    if config is None or not getattr(config, 'enable_candidate_diag_log', False):
        return
    log_path = getattr(config, 'candidate_diag_log_path', None)
    if not log_path:
        return
    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        seq_id = getattr(config, 'current_seq_id', 'unknown')
        curr_frame = getattr(config, 'current_data_frame', None)
        curr_frame = -1 if curr_frame is None else int(curr_frame)
        track_id = int(getattr(track, 'track_id_3d', -1))
        det_score = float(getattr(detection, 'score', 1.0))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                '[L2.5 CandidateDiag][{}] frame={} track_id={} det_frame={} dt={} stage={} decision={} reason={} '
                'det_score={:.6f} iou={} app_sim={} vel_sim={} decay={} uncertainty={} motion_reliability={} '
                'w_iou={} w_vel={} w_app={} vel_share={} vel_focus={} combined_sim={} cost={}\n'.format(
                    seq_id,
                    curr_frame,
                    track_id,
                    int(detection_frame_id),
                    int(dt),
                    str(stage),
                    str(decision),
                    str(reason),
                    det_score,
                    'nan' if iou is None else '{:.6f}'.format(float(iou)),
                    'nan' if app_sim is None else '{:.6f}'.format(float(app_sim)),
                    'nan' if vel_sim is None else '{:.6f}'.format(float(vel_sim)),
                    'nan' if decay is None else '{:.6f}'.format(float(decay)),
                    'nan' if uncertainty is None else '{:.6f}'.format(float(uncertainty)),
                    'nan' if motion_reliability is None else '{:.6f}'.format(float(motion_reliability)),
                    'nan' if w_iou is None else '{:.6f}'.format(float(w_iou)),
                    'nan' if w_vel is None else '{:.6f}'.format(float(w_vel)),
                    'nan' if w_app is None else '{:.6f}'.format(float(w_app)),
                    'nan' if vel_share is None else '{:.6f}'.format(float(vel_share)),
                    'nan' if vel_focus is None else '{:.6f}'.format(float(vel_focus)),
                    'nan' if combined_sim is None else '{:.6f}'.format(float(combined_sim)),
                    'nan' if cost is None else '{:.6f}'.format(float(cost)),
                )
            )
    except Exception:
        pass

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


def _safe_diag_values(matrix, limit=10):
    try:
        diag = np.diag(matrix).reshape(-1)
        return [float(x) for x in diag[:limit]]
    except Exception:
        return []


def _append_covariance_diag_log(config, track, detection_frame_id, current_frame, dt,
                                decay, diag_before, diag_after_update, diag_after_fast_forward):
    if config is None or not getattr(config, 'enable_covariance_diag_log', False):
        return

    log_path = getattr(config, 'covariance_diag_log_path', None)
    if not log_path:
        return

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        seq_id = getattr(config, 'current_seq_id', 'unknown')
        curr_frame = getattr(config, 'current_data_frame', None)
        curr_frame = -1 if curr_frame is None else int(curr_frame)
        track_id = int(getattr(track, 'track_id_3d', -1))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                '[L2.5 CovDiag] seq={} frame={} track_id={} det_frame={} dt={} decay={:.6f} '
                'diag_before={} diag_after_update={} diag_after_fast_forward={}\n'.format(
                    seq_id,
                    curr_frame,
                    track_id,
                    int(detection_frame_id),
                    int(dt),
                    float(decay),
                    diag_before,
                    diag_after_update,
                    diag_after_fast_forward,
                )
            )
    except Exception:
        pass


def _append_l25_hit_event_log(config, track, detection_frame_id, current_frame, dt, decay, time_since_update):
    if config is None or not getattr(config, 'enable_hit_event_log', False):
        return

    log_path = getattr(config, 'hit_event_log_path', None)
    if not log_path:
        return

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        seq_id = getattr(config, 'current_seq_id', 'unknown')
        curr_frame = getattr(config, 'current_data_frame', None)
        curr_frame = -1 if curr_frame is None else int(curr_frame)
        track_id = int(getattr(track, 'track_id_3d', -1))
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(
                '[L2.5 Hit] seq={} frame={} track_id={} det_frame={} dt={} decay={:.6f} tsu={}\n'.format(
                    seq_id,
                    curr_frame,
                    track_id,
                    int(detection_frame_id),
                    int(dt),
                    float(decay),
                    int(time_since_update),
                )
            )
    except Exception:
        pass


def get_pose_at_past_frame(track, time_diff, use_nonlinear=True, verbose=False, config=None):
    """
    基于当前KF状态，将轨迹回推 time_diff 帧，返回回推后的7维pose，不修改原状态。
    
    Args:
        track: 轨迹对象
        time_diff: 回推的帧数
        use_nonlinear: 是否使用非线性回推（考虑加速度）
        verbose: 是否输出调试信息
        config: MultiFrameBacktrackConfig 配置对象
    
    Returns:
        pose: 7维pose [x, y, z, theta, l, w, h]
    """
    try:
        x = track.kf_3d.kf.x.copy()
        current_pos = x[:3].reshape(3)
        theta = float(x[3])
        size = x[4:7].reshape(3)
        
        # 加速度门控：只在加速度显著时使用非线性
        use_acceleration_gate = getattr(config, 'use_acceleration_gate', False) if config else False
        acceleration_threshold = getattr(config, 'acceleration_threshold', 1.5) if config else 1.5
        
        # 方法2：使用历史时刻的速度和加速度（更物理准确）
        if use_nonlinear and hasattr(track, 'velocity_history') and len(track.velocity_history) >= 2:
            current_frame = track.velocity_history[-1][0] if len(track.velocity_history) > 0 else 0
            target_frame = current_frame - time_diff
            
            # 从速度历史中查找目标帧附近的速度
            v_at_target = None
            a_at_target = None
            
            # 查找最接近目标帧的速度记录
            for i in range(len(track.velocity_history) - 1, -1, -1):
                frame_id, vel = track.velocity_history[i]
                if frame_id <= target_frame:
                    v_at_target = vel
                    # 计算该时刻的加速度（如果有前一个速度点）
                    if i > 0:
                        prev_frame, prev_vel = track.velocity_history[i-1]
                        dt = max(frame_id - prev_frame, 1)
                        a_at_target = (vel - prev_vel) / dt
                    break
            
            # 如果找到了历史速度，使用历史速度回推
            if v_at_target is not None:
                dt = float(time_diff)
                v_current = x[7:10].reshape(3)
                
                # 加速度门控判断
                use_nonlinear_for_this = True
                if use_acceleration_gate and a_at_target is not None:
                    a_norm = float(np.linalg.norm(a_at_target))
                    if a_norm < acceleration_threshold:
                        # 加速度不显著，使用线性回推
                        use_nonlinear_for_this = False
                        if verbose and np.random.rand() < 0.01:
                            print(f"[加速度门控] |a|={a_norm:.3f} < {acceleration_threshold:.3f}, 使用线性回推")
                    else:
                        if verbose and np.random.rand() < 0.01:
                            print(f"[加速度门控] |a|={a_norm:.3f} >= {acceleration_threshold:.3f}, 使用非线性回推")
                
                if verbose and np.random.rand() < 0.01:  # 1%概率输出，避免刷屏
                    v_norm_target = float(np.linalg.norm(v_at_target))
                    v_norm_current = float(np.linalg.norm(v_current))
                    print(f"[历史速度回推] Δt={dt:.0f}, v_target={v_norm_target:.2f}, v_current={v_norm_current:.2f}, 差异={abs(v_norm_target-v_norm_current):.2f}")
                
                if use_nonlinear_for_this and a_at_target is not None:
                    # 使用历史时刻的速度和加速度（非线性）
                    pos = current_pos - v_at_target * dt - 0.5 * a_at_target * (dt ** 2)
                else:
                    # 只有速度，没有加速度，或加速度不显著（线性）
                    pos = current_pos - v_at_target * dt
            else:
                # 没找到历史速度，降级到使用当前速度
                v_current = x[7:10].reshape(3)
                pos = current_pos - v_current * float(time_diff)
        else:
            # 线性回推（原始方法）
            v_current = x[7:10].reshape(3)
            pos = current_pos - v_current * float(time_diff)
        
        pose = np.zeros(7, dtype=np.float32)
        pose[0:3] = pos
        pose[3] = theta
        pose[4:7] = size
        return pose
    except Exception as e:
        # 降级到简单线性回推
        v = get_velocity(track)
        pos = track.pose[:3] - v[:3] * float(time_diff)
        pose = np.zeros(7, dtype=np.float32)
        pose[0:3] = pos
        pose[3] = track.pose[3]
        pose[4:7] = track.pose[4:7]
        return pose

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


def _get_l25_geometry_mode(config):
    mode = str(getattr(config, 'geometry_mode', 'box_iou')).strip().lower()
    if mode in ('rotated', 'rotated_geom', 'rot_geom'):
        return 'rotated_geom'
    return 'box_iou'


def _compute_l25_geometry_similarity(rollback_pose, det_bbox, config):
    geom_mode = _get_l25_geometry_mode(config)
    box_iou = compute_iou_3d(rollback_pose, det_bbox)

    if geom_mode == 'rotated_geom':
        geom_sim = compute_rotated_ground_similarity(det_bbox, rollback_pose)
        return float(np.clip(geom_sim, 0.0, 1.0)), float(np.clip(box_iou, 0.0, 1.0)), geom_mode

    geom_sim = box_iou
    if getattr(config, 'use_rotated_geom_in_l25', False):
        rotated_geom_sim = compute_rotated_ground_similarity(det_bbox, rollback_pose)
        geom_w = max(0.0, min(1.0, float(getattr(config, 'rotated_geom_weight_l25', 0.10))))
        geom_sim = (1.0 - geom_w) * geom_sim + geom_w * rotated_geom_sim
    return float(np.clip(geom_sim, 0.0, 1.0)), float(np.clip(box_iou, 0.0, 1.0)), geom_mode


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
    # 简化版: 如果有嵌入向量，计算余弦相似度（鲁棒化处理维度不一致）
    if hasattr(track, 'emb') and hasattr(detection, 'feature') and \
       track.emb is not None and detection.feature is not None:
        try:
            te = np.asarray(track.emb).reshape(-1)
            de = np.asarray(detection.feature).reshape(-1)
            # 若维度不一致，按最小维度对齐；过短时直接退化
            min_dim = int(min(te.size, de.size))
            if min_dim <= 1:
                return 0.5
            te = te[:min_dim]
            de = de[:min_dim]
            # 归一化
            te = te / (np.linalg.norm(te) + 1e-6)
            de = de / (np.linalg.norm(de) + 1e-6)
            sim = float(np.dot(te, de))
            # 将[-1,1]裁剪并线性映射到[0,1]
            sim = max(-1.0, min(1.0, sim))
            return 0.5 * (sim + 1.0)
        except Exception:
            return 0.5
    
    return 0.5  # 默认中等相似度


def compute_memory_bank_appearance_similarity(track, detection):
    if not hasattr(detection, 'feature') or detection.feature is None:
        return compute_appearance_similarity(track, detection)
    if not hasattr(track, 'appearance_memory_bank') or len(track.appearance_memory_bank) == 0:
        return compute_appearance_similarity(track, detection)

    try:
        de = np.asarray(detection.feature).reshape(-1)
        if de.size <= 1:
            return compute_appearance_similarity(track, detection)
        de = de / (np.linalg.norm(de) + 1e-6)
        best_sim = None
        for mem in track.appearance_memory_bank:
            me = np.asarray(mem).reshape(-1)
            min_dim = int(min(me.size, de.size))
            if min_dim <= 1:
                continue
            me_use = me[:min_dim]
            de_use = de[:min_dim]
            me_use = me_use / (np.linalg.norm(me_use) + 1e-6)
            de_use = de_use / (np.linalg.norm(de_use) + 1e-6)
            sim = float(np.dot(me_use, de_use))
            sim = max(-1.0, min(1.0, sim))
            sim = 0.5 * (sim + 1.0)
            if best_sim is None or sim > best_sim:
                best_sim = sim
        if best_sim is None:
            return compute_appearance_similarity(track, detection)
        return float(best_sim)
    except Exception:
        return compute_appearance_similarity(track, detection)


def compute_memory_bank_appearance_details(track, detection):
    base_sim = compute_appearance_similarity(track, detection)
    if not hasattr(detection, 'feature') or detection.feature is None:
        return float(base_sim), float(base_sim), False
    if not hasattr(track, 'appearance_memory_bank') or len(track.appearance_memory_bank) == 0:
        return float(base_sim), float(base_sim), False
    mem_sim = compute_memory_bank_appearance_similarity(track, detection)
    return float(mem_sim), float(base_sim), True


def _compute_l25_uncertainty(track, config):
    try:
        P = track.kf_3d.kf.P
        sx = float(np.sqrt(np.abs(P[0, 0]))) if P.shape[0] > 0 else 0.0
        sz = float(np.sqrt(np.abs(P[2, 2]))) if P.shape[0] > 2 else 0.0
        raw_uncertainty = max(0.0, sx + sz)
        u_norm = max(float(getattr(config, 'uncertainty_norm', 12.0)), 1e-6)
        uncertainty = 1.0 - math.exp(-raw_uncertainty / u_norm)
        return max(0.0, min(1.0, float(uncertainty)))
    except Exception:
        return 0.0


def _safe_sigmoid(x):
    x = np.clip(float(x), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


def _get_l25_motion_reliability_calibrator(config):
    if config is None:
        return None
    signature = (
        bool(getattr(config, 'enable_confidence_aware_motion_l25', False)),
        str(getattr(config, 'l25_motion_reliability_mode', 'manual')).strip().lower(),
        str(getattr(config, 'l25_motion_reliability_model_path', '') or '').strip(),
        float(getattr(config, 'l25_motion_reliability_bias', -0.1)),
        float(getattr(config, 'l25_motion_reliability_score_gain', 2.0)),
        float(getattr(config, 'l25_motion_reliability_uncertainty_gain', 2.0)),
        tuple(getattr(config, 'l25_motion_reliability_feature_names', MotionReliabilityCalibrator.DEFAULT_FEATURES)),
    )
    if getattr(config, '_l25_motion_reliability_calibrator', None) is not None and getattr(config, '_l25_motion_reliability_signature', None) == signature:
        return config._l25_motion_reliability_calibrator
    config._l25_motion_reliability_calibrator = MotionReliabilityCalibrator(
        mode=getattr(config, 'l25_motion_reliability_mode', 'manual'),
        manual_bias=getattr(config, 'l25_motion_reliability_bias', -0.1),
        manual_score_gain=getattr(config, 'l25_motion_reliability_score_gain', 2.0),
        manual_uncertainty_gain=getattr(config, 'l25_motion_reliability_uncertainty_gain', 2.0),
        feature_names=getattr(config, 'l25_motion_reliability_feature_names', MotionReliabilityCalibrator.DEFAULT_FEATURES),
        weight_path=getattr(config, 'l25_motion_reliability_model_path', ''),
    )
    config._l25_motion_reliability_signature = signature
    return config._l25_motion_reliability_calibrator


def _compute_l25_motion_reliability(track, detection, rollback_pose, vel_sim, uncertainty, config):
    if config is None or not getattr(config, 'enable_confidence_aware_motion_l25', False):
        return 1.0

    try:
        det_score = _clip01(getattr(detection, 'score', 1.0))
    except Exception:
        det_score = 1.0
    try:
        track_tsu = _clip01(float(getattr(track, 'time_since_update', 0)) / max(float(getattr(config, 'max_backtrack_age', 15)), 1.0))
    except Exception:
        track_tsu = 0.0
    try:
        track_beta = _clip01(float(getattr(track, 'beta_t', 1.0)))
    except Exception:
        track_beta = 1.0
    try:
        track_hits = _clip01(float(getattr(track, 'hits', 0)) / 8.0)
    except Exception:
        track_hits = 0.0
    try:
        track_age = _clip01(float(getattr(track, 'age', 0)) / 30.0)
    except Exception:
        track_age = 0.0
    try:
        track_speed = _clip01(float(np.linalg.norm(np.asarray(get_velocity(track), dtype=np.float32))) / max(float(getattr(config, 'vmax_for_adaptive_weight', 12.0)), 1e-6))
    except Exception:
        track_speed = 0.0

    pair_center_dist = 0.0
    try:
        det_box = np.asarray(detection.bbox, dtype=np.float32)
        trk_box = np.asarray(rollback_pose, dtype=np.float32)
        center_dist = float(np.linalg.norm(det_box[[0, 2]] - trk_box[[0, 2]]))
        det_diag = float(np.hypot(max(det_box[4], 1e-3), max(det_box[5], 1e-3)))
        trk_diag = float(np.hypot(max(trk_box[4], 1e-3), max(trk_box[5], 1e-3)))
        scale_ref = max(0.5 * (det_diag + trk_diag), 1e-3)
        pair_center_dist = _clip01(center_dist / (2.5 * scale_ref))
    except Exception:
        pair_center_dist = 0.0

    features = MotionReliabilityCalibrator.build_feature_dict(
        det_score=det_score,
        track_uncertainty=uncertainty,
        track_tsu=track_tsu,
        track_beta=track_beta,
        track_hits=track_hits,
        track_age=track_age,
        track_speed=track_speed,
        pair_center_dist=pair_center_dist,
        vel_sim=vel_sim,
    )
    calibrator = _get_l25_motion_reliability_calibrator(config)
    if calibrator is None:
        return 1.0
    return _clip01(calibrator.predict(features))


def _compute_l25_weight_terms(track, geom_sim, vel_sim, app_sim, motion_reliability, config, w_vel_t, uncertainty):
    cap = 0.25 if getattr(track, 'time_since_update', 0) >= 3 else 0.22
    base_app = min(max(getattr(config, 'appearance_weight', 0.2), 0.0), cap)
    reliable = (app_sim >= 0.6)
    w_app = base_app if reliable else min(0.05, base_app)
    residual = max(0.0, 1.0 - w_app)

    state_floor = _clip01(getattr(config, 'velocity_confidence_floor', 0.25))
    state_conf = _clip01(state_floor + (1.0 - state_floor) * (1.0 - _clip01(uncertainty)))

    geom_focus = _safe_sigmoid(
        float(getattr(config, 'l25_velocity_geom_gain', 10.0)) *
        (_clip01(geom_sim) - float(getattr(config, 'l25_velocity_geom_center', 0.30)))
    )
    motion_focus = _safe_sigmoid(
        float(getattr(config, 'l25_velocity_motion_gain', 8.0)) *
        (_clip01(vel_sim) - float(getattr(config, 'l25_velocity_motion_center', 0.55)))
    )
    vel_focus = math.sqrt(max(0.0, geom_focus * motion_focus))
    speed_prior = _clip01(w_vel_t)
    candidate_value = _clip01(0.50 * vel_focus + 0.35 * speed_prior + 0.15 * _clip01(geom_sim))

    vel_share_min = _clip01(getattr(config, 'l25_velocity_share_min', 0.05))
    vel_share_max = max(vel_share_min, _clip01(getattr(config, 'l25_velocity_share_max', 0.38)))
    base_vel_share = vel_share_min + (vel_share_max - vel_share_min) * candidate_value

    reliability_strength = _clip01(getattr(config, 'l25_velocity_reliability_strength', 0.85))
    reliability_gate = _clip01((1.0 - reliability_strength) + reliability_strength * _clip01(motion_reliability))
    effective_vel_share = _clip01(base_vel_share * math.sqrt(max(0.0, state_conf * reliability_gate)))

    w_vel = min(residual * effective_vel_share, residual)
    w_iou = max(0.0, residual - w_vel)
    return float(w_iou), float(w_vel), float(w_app), float(effective_vel_share), float(vel_focus)


def _build_l25_cost(iou, vel_sim, app_sim, motion_reliability, track, config, w_vel_t, uncertainty, decay):
    w_iou, w_vel, w_app, _, _ = _compute_l25_weight_terms(
        track, iou, vel_sim, app_sim, motion_reliability, config, w_vel_t, uncertainty
    )
    combined_sim = w_iou * iou + w_vel * vel_sim + w_app * app_sim
    return float(-(combined_sim * decay))

def _passes_l25_candidate_pre_gate(rollback_pose, det_bbox, iou, dt, config):
    if config is None or not getattr(config, 'enable_candidate_pre_gate', True):
        return True

    min_iou = float(getattr(config, 'candidate_min_iou', 0.03))
    if float(iou) < min_iou:
        return False

    try:
        trk_center = np.asarray(rollback_pose[:3], dtype=np.float32)
        det_center = np.asarray(det_bbox[:3], dtype=np.float32)
        center_dist = float(np.linalg.norm(trk_center[[0, 2]] - det_center[[0, 2]]))
    except Exception:
        center_dist = float('inf')

    max_center_dist = (
        float(getattr(config, 'candidate_max_center_dist_base', 2.0)) +
        max(0.0, float(dt) - 1.0) * float(getattr(config, 'candidate_max_center_dist_per_dt', 0.35))
    )
    if center_dist > max_center_dist:
        return False

    try:
        trk_size = np.maximum(np.asarray(rollback_pose[4:7], dtype=np.float32), 1e-6)
        det_size = np.maximum(np.asarray(det_bbox[4:7], dtype=np.float32), 1e-6)
        size_ratio = float(np.min(np.minimum(trk_size, det_size) / np.maximum(trk_size, det_size)))
    except Exception:
        size_ratio = 0.0

    min_size_ratio = float(getattr(config, 'candidate_min_size_ratio', 0.55))
    if size_ratio < min_size_ratio:
        return False

    return True


def _get_l25_candidate_pre_gate_failure_reason(rollback_pose, det_bbox, iou, dt, config):
    if config is None or not getattr(config, 'enable_candidate_pre_gate', True):
        return None

    min_iou = float(getattr(config, 'candidate_min_iou', 0.03))
    if float(iou) < min_iou:
        return 'pre_gate_iou'

    try:
        trk_center = np.asarray(rollback_pose[:3], dtype=np.float32)
        det_center = np.asarray(det_bbox[:3], dtype=np.float32)
        center_dist = float(np.linalg.norm(trk_center[[0, 2]] - det_center[[0, 2]]))
    except Exception:
        center_dist = float('inf')

    max_center_dist = (
        float(getattr(config, 'candidate_max_center_dist_base', 2.0)) +
        max(0.0, float(dt) - 1.0) * float(getattr(config, 'candidate_max_center_dist_per_dt', 0.35))
    )
    if center_dist > max_center_dist:
        return 'pre_gate_center_dist'

    try:
        trk_size = np.maximum(np.asarray(rollback_pose[4:7], dtype=np.float32), 1e-6)
        det_size = np.maximum(np.asarray(det_bbox[4:7], dtype=np.float32), 1e-6)
        size_ratio = float(np.min(np.minimum(trk_size, det_size) / np.maximum(trk_size, det_size)))
    except Exception:
        size_ratio = 0.0

    min_size_ratio = float(getattr(config, 'candidate_min_size_ratio', 0.55))
    if size_ratio < min_size_ratio:
        return 'pre_gate_size_ratio'

    return None


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
    if config is None:
        config = MultiFrameBacktrackConfig()
    # 仅考虑最近K帧，按时间就近优先
    allowed_dts = getattr(config, 'allowed_backtrack_dts', None)
    frames = []
    for fid in detection_buffer.keys():
        dt = current_frame - fid
        if dt > 0 and dt <= getattr(config, 'last_k_frames', 5):
            if allowed_dts is not None and dt not in allowed_dts:
                continue
            frames.append((dt, fid))
    frames.sort(key=lambda x: x[0])  # t-1, t-2, ...

    for dt, fid in frames:
        detections = detection_buffer.get(fid, [])
        per_frame = []
        for det in detections:
            # 相似度项（不使用角/角速度）
            # 方案A：将轨迹回推到历史帧，与历史检测计算IoU
            use_nonlinear = getattr(config, 'use_nonlinear_backtrack', True)
            rollback_pose = get_pose_at_past_frame(track, dt, use_nonlinear=use_nonlinear, config=config)
            decay = compute_decay_factor(dt, config.lambda_decay)
            geom_sim, gate_iou, _ = _compute_l25_geometry_similarity(rollback_pose, det.bbox, config)
            if gate_iou <= 1e-6:
                _append_l25_candidate_diag_log(config, track, det, fid, dt, 'single', 'reject', 'iou_zero', iou=geom_sim, decay=decay)
                continue
            pre_gate_reason = _get_l25_candidate_pre_gate_failure_reason(rollback_pose, det.bbox, gate_iou, dt, config)
            if pre_gate_reason is not None:
                _append_l25_candidate_diag_log(config, track, det, fid, dt, 'single', 'reject', pre_gate_reason, iou=geom_sim, decay=decay)
                continue
            if getattr(config, 'use_l25_memory_bank_appearance', False):
                app_sim = compute_memory_bank_appearance_similarity(track, det)
            else:
                app_sim = compute_appearance_similarity(track, det)
            if app_sim < getattr(config, 'appearance_hard_gate', 0.6):
                _append_l25_candidate_diag_log(config, track, det, fid, dt, 'single', 'reject', 'appearance_hard_gate', iou=geom_sim, app_sim=app_sim, decay=decay)
                continue
            det_vel = estimate_detection_velocity(det, detection_buffer, fid)
            trk_vel = get_velocity(track)
            vel_sim = compute_velocity_similarity_vec(trk_vel, det_vel)
            # 外观上限：短遮挡/长遮挡
            cap = 0.25 if getattr(track, 'time_since_update', 0) >= 3 else 0.22
            base_app = min(max(getattr(config, 'appearance_weight', 0.2), 0.0), cap)
            # 轻量可靠性判据：外观相似度阈值
            reliable = (app_sim >= 0.6)
            w_app = base_app if reliable else min(0.05, base_app)
            residual = max(0.0, 1.0 - w_app)
            vmax = getattr(config, 'vmax_for_adaptive_weight', 10.0)
            w_vel_t, _ = compute_adaptive_weight_linear(get_velocity(track), v_max=vmax)
            uncertainty = _compute_l25_uncertainty(track, config)
            motion_reliability = _compute_l25_motion_reliability(
                track, det, rollback_pose, vel_sim, uncertainty, config
            )
            w_iou, w_vel, w_app, vel_share, vel_focus = _compute_l25_weight_terms(
                track, geom_sim, vel_sim, app_sim, motion_reliability, config, w_vel_t, uncertainty
            )
            combined_sim = w_iou * geom_sim + w_vel * vel_sim + w_app * app_sim
            decayed_sim = combined_sim * decay
            cost = -decayed_sim
            _append_l25_candidate_diag_log(
                config, track, det, fid, dt, 'single', 'keep', 'candidate',
                iou=geom_sim, app_sim=app_sim, vel_sim=vel_sim, decay=decay,
                uncertainty=uncertainty, motion_reliability=motion_reliability,
                w_iou=w_iou, w_vel=w_vel, w_app=w_app, vel_share=vel_share, vel_focus=vel_focus,
                combined_sim=combined_sim, cost=cost,
            )
            per_frame.append((fid, det, cost, dt, decay))
        # 帧内取top-k
        if len(per_frame) > 0:
            per_frame.sort(key=lambda x: x[2])
            k = getattr(config, 'topk_per_frame', 2)
            k = 2 if (k is None or k <= 0) else int(k)
            candidates.extend(per_frame[:k])
    return candidates


def multi_frame_backtrack_association(unmatched_tracks, detection_buffer, 
                                     current_frame, config=None):
    """
    ??????
    
    Args:
        unmatched_tracks: ???????
        detection_buffer: ???? {frame_id: [detections]}
        current_frame: ???
        config: MultiFrameBacktrackConfig??
    
    Returns:
        matched_pairs: [(track, detection, detection_frame_id), ...]
    """
    if config is None:
        config = MultiFrameBacktrackConfig()

    if not config.enable_multi_frame_backtrack:
        return []

    matched_pairs = []
    used_detections = set()
    cooldown_enabled = getattr(config, 'enable_l25_cooldown', False)
    cooldown_frames = max(0, int(getattr(config, 'l25_cooldown_frames', 0)))
    allowed_dts = getattr(config, 'allowed_backtrack_dts', None)
    memory_bank_stats = _ensure_memory_bank_stats(config)
    if memory_bank_stats is not None:
        memory_bank_stats['calls'] += 1

    if getattr(config, 'use_global_assignment', False):
        frames = []
        for fid in detection_buffer.keys():
            dt = current_frame - fid
            if dt > 0 and dt <= getattr(config, 'last_k_frames', 5):
                if allowed_dts is not None and dt not in allowed_dts:
                    continue
                frames.append((dt, fid))
        frames.sort(key=lambda x: x[0])

        cand_list = []
        seen = set()
        for dt, fid in frames:
            for det in detection_buffer.get(fid, []):
                det_id = id(det)
                if det_id in seen:
                    continue
                seen.add(det_id)
                cand_list.append((fid, det, dt, None))

        if len(cand_list) == 0 or len(unmatched_tracks) == 0:
            return matched_pairs

        INF = 1e6
        use_memory_bank = bool(getattr(config, 'use_l25_memory_bank_appearance', False))
        C_base = np.full((len(cand_list), len(unmatched_tracks)), INF, dtype=np.float32)
        C = np.full((len(cand_list), len(unmatched_tracks)), INF, dtype=np.float32)
        mem_costs = np.full((len(cand_list), len(unmatched_tracks)), INF, dtype=np.float32) if use_memory_bank else None
        mem_valid = np.zeros((len(cand_list), len(unmatched_tracks)), dtype=bool) if use_memory_bank else None

        for j, (fid, det, dt, _) in enumerate(cand_list):
            for i, track in enumerate(unmatched_tracks):
                if cooldown_enabled:
                    last_l25_frame = getattr(track, 'last_l25_recovery_frame', None)
                    if last_l25_frame is not None and (current_frame - int(last_l25_frame)) <= cooldown_frames:
                        continue
                if not (config.min_backtrack_age <= track.time_since_update <= config.max_backtrack_age):
                    continue

                use_nonlinear = getattr(config, 'use_nonlinear_backtrack', True)
                rollback_pose = get_pose_at_past_frame(track, dt, use_nonlinear=use_nonlinear, config=config)
                decay = compute_decay_factor(dt, config.lambda_decay)
                geom_sim, gate_iou, _ = _compute_l25_geometry_similarity(rollback_pose, det.bbox, config)
                if gate_iou <= 1e-6:
                    _append_l25_candidate_diag_log(config, track, det, fid, dt, 'global', 'reject', 'iou_zero', iou=geom_sim, decay=decay)
                    continue
                pre_gate_reason = _get_l25_candidate_pre_gate_failure_reason(rollback_pose, det.bbox, gate_iou, dt, config)
                if pre_gate_reason is not None:
                    _append_l25_candidate_diag_log(config, track, det, fid, dt, 'global', 'reject', pre_gate_reason, iou=geom_sim, decay=decay)
                    continue

                base_app_sim = compute_appearance_similarity(track, det)
                if base_app_sim < getattr(config, 'appearance_hard_gate', 0.6):
                    _append_l25_candidate_diag_log(config, track, det, fid, dt, 'global', 'reject', 'appearance_hard_gate', iou=geom_sim, app_sim=base_app_sim, decay=decay)
                    continue

                det_vel = estimate_detection_velocity(det, detection_buffer, fid)
                trk_vel = get_velocity(track)
                vel_sim = compute_velocity_similarity_vec(trk_vel, det_vel)
                vmax = getattr(config, 'vmax_for_adaptive_weight', 10.0)
                w_vel_t, _ = compute_adaptive_weight_linear(get_velocity(track), v_max=vmax)
                uncertainty = _compute_l25_uncertainty(track, config)
                motion_reliability = _compute_l25_motion_reliability(
                    track, det, rollback_pose, vel_sim, uncertainty, config
                )

                base_cost = _build_l25_cost(
                    iou=geom_sim,
                    vel_sim=vel_sim,
                    app_sim=float(base_app_sim),
                    motion_reliability=motion_reliability,
                    track=track,
                    config=config,
                    w_vel_t=w_vel_t,
                    uncertainty=uncertainty,
                    decay=decay,
                )
                w_iou, w_vel, w_app, vel_share, vel_focus = _compute_l25_weight_terms(
                    track, geom_sim, vel_sim, base_app_sim, motion_reliability, config, w_vel_t, uncertainty
                )
                combined_sim = w_iou * geom_sim + w_vel * vel_sim + w_app * base_app_sim
                if base_cost >= config.cost_threshold:
                    _append_l25_candidate_diag_log(
                        config, track, det, fid, dt, 'global', 'reject', 'cost_threshold',
                        iou=geom_sim, app_sim=base_app_sim, vel_sim=vel_sim, decay=decay,
                        uncertainty=uncertainty, motion_reliability=motion_reliability,
                        w_iou=w_iou, w_vel=w_vel, w_app=w_app, vel_share=vel_share, vel_focus=vel_focus,
                        combined_sim=combined_sim, cost=base_cost,
                    )
                    continue

                _append_l25_candidate_diag_log(
                    config, track, det, fid, dt, 'global', 'keep', 'cost_valid',
                    iou=geom_sim, app_sim=base_app_sim, vel_sim=vel_sim, decay=decay,
                    uncertainty=uncertainty, motion_reliability=motion_reliability,
                    w_iou=w_iou, w_vel=w_vel, w_app=w_app, vel_share=vel_share, vel_focus=vel_focus,
                    combined_sim=combined_sim, cost=base_cost,
                )

                C_base[j, i] = base_cost
                C[j, i] = base_cost

                if memory_bank_stats is not None:
                    memory_bank_stats['pairs_considered'] += 1

                if use_memory_bank:
                    mem_app_sim, base_app_dup, used_memory_bank_pair = compute_memory_bank_appearance_details(track, det)
                    if memory_bank_stats is not None and used_memory_bank_pair:
                        memory_bank_stats['pairs_with_memory_bank'] += 1
                        if abs(float(mem_app_sim) - float(base_app_dup)) > 1e-6:
                            memory_bank_stats['pairs_app_changed'] += 1
                    if used_memory_bank_pair and mem_app_sim >= getattr(config, 'appearance_hard_gate', 0.6):
                        mem_cost = _build_l25_cost(
                            iou=geom_sim,
                            vel_sim=vel_sim,
                            app_sim=float(mem_app_sim),
                            motion_reliability=motion_reliability,
                            track=track,
                            config=config,
                            w_vel_t=w_vel_t,
                            uncertainty=uncertainty,
                            decay=decay,
                        )
                        if mem_cost < config.cost_threshold:
                            mem_costs[j, i] = mem_cost
                            mem_valid[j, i] = True
                            if memory_bank_stats is not None and abs(float(mem_cost) - float(base_cost)) > 1e-6:
                                memory_bank_stats['pairs_cost_changed'] += 1

        if use_memory_bank and mem_costs is not None and mem_valid is not None:
            margin = float(getattr(config, 'memory_bank_rescore_margin', 0.03))
            ambiguous_rows = set()
            ambiguous_cols = set()

            for row_idx in range(C_base.shape[0]):
                finite_vals = C_base[row_idx][C_base[row_idx] < INF]
                if finite_vals.size >= 2:
                    sorted_vals = np.sort(finite_vals)
                    if float(sorted_vals[1] - sorted_vals[0]) <= margin:
                        ambiguous_rows.add(row_idx)

            for col_idx in range(C_base.shape[1]):
                finite_vals = C_base[:, col_idx][C_base[:, col_idx] < INF]
                if finite_vals.size >= 2:
                    sorted_vals = np.sort(finite_vals)
                    if float(sorted_vals[1] - sorted_vals[0]) <= margin:
                        ambiguous_cols.add(col_idx)

            if memory_bank_stats is not None:
                memory_bank_stats['ambiguous_rows'] += len(ambiguous_rows)
                memory_bank_stats['ambiguous_cols'] += len(ambiguous_cols)

            rescored_pairs = set()
            for row_idx in ambiguous_rows:
                for col_idx in np.where(mem_valid[row_idx])[0]:
                    C[row_idx, col_idx] = mem_costs[row_idx, col_idx]
                    rescored_pairs.add((int(row_idx), int(col_idx)))
            for col_idx in ambiguous_cols:
                for row_idx in np.where(mem_valid[:, col_idx])[0]:
                    C[row_idx, col_idx] = mem_costs[row_idx, col_idx]
                    rescored_pairs.add((int(row_idx), int(col_idx)))

            if memory_bank_stats is not None:
                memory_bank_stats['rescored_pairs'] += len(rescored_pairs)

        assign = linear_assignment(C)
        if use_memory_bank and memory_bank_stats is not None:
            memory_bank_stats['assignment_calls'] += 1
            assign_base = linear_assignment(C_base)
            bank_set = set()
            base_set = set()
            if getattr(assign, 'size', 0) != 0:
                for row_idx, col_idx in assign:
                    if row_idx < 0 or row_idx >= C.shape[0] or col_idx < 0 or col_idx >= C.shape[1]:
                        continue
                    pair_cost = float(C[row_idx, col_idx])
                    if np.isfinite(pair_cost) and pair_cost < config.cost_threshold:
                        bank_set.add((int(row_idx), int(col_idx)))
            if getattr(assign_base, 'size', 0) != 0:
                for row_idx, col_idx in assign_base:
                    if row_idx < 0 or row_idx >= C_base.shape[0] or col_idx < 0 or col_idx >= C_base.shape[1]:
                        continue
                    pair_cost = float(C_base[row_idx, col_idx])
                    if np.isfinite(pair_cost) and pair_cost < config.cost_threshold:
                        base_set.add((int(row_idx), int(col_idx)))
            diff_pairs = bank_set.symmetric_difference(base_set)
            if len(diff_pairs) > 0:
                memory_bank_stats['assignment_changed_calls'] += 1
                memory_bank_stats['assignment_changed_pairs'] += len(diff_pairs)

        if assign.size == 0:
            return matched_pairs

        for row_idx, col_idx in assign:
            if row_idx < 0 or row_idx >= C.shape[0] or col_idx < 0 or col_idx >= C.shape[1]:
                continue
            pair_cost = float(C[row_idx, col_idx])
            if not np.isfinite(pair_cost) or pair_cost >= config.cost_threshold:
                continue
            fid, det, dt, _ = cand_list[int(row_idx)]
            track = unmatched_tracks[int(col_idx)]
            decay = compute_decay_factor(dt, config.lambda_decay)
            matched_pairs.append((track, det, fid, dt, decay))
            if config.verbose:
                print(f"[????-GLOBAL] track={track.track_id_3d} fid={fid} dt={dt} cost={pair_cost:.4f}")

        return matched_pairs

    for track in unmatched_tracks:
        if cooldown_enabled:
            last_l25_frame = getattr(track, 'last_l25_recovery_frame', None)
            if last_l25_frame is not None and (current_frame - int(last_l25_frame)) <= cooldown_frames:
                continue
        if not (config.min_backtrack_age <= track.time_since_update <= config.max_backtrack_age):
            continue
        candidates = compute_decay_cost_matrix(
            track, detection_buffer, current_frame, config
        )
        if not candidates:
            continue
        selected = None
        for fid, det, cst, time_diff, decay in candidates:
            if cst < config.cost_threshold and (id(det) not in used_detections):
                selected = (fid, det, cst, time_diff, decay)
                break
        if selected is None:
            continue
        best_frame_id, best_det, best_cost, time_diff, decay = selected
        matched_pairs.append((track, best_det, best_frame_id, time_diff, decay))
        used_detections.add(id(best_det))
        if config.verbose:
            print(f"[????] track={track.track_id_3d} frame={best_frame_id} dt={time_diff} cost={best_cost:.4f}")
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
        dt = int(max(0, time_diff))
        time_since_update_before = int(getattr(track, 'time_since_update', -1))
        x_backup = track.kf_3d.kf.x.copy()
        P_backup = track.kf_3d.kf.P.copy()
        diag_before = _safe_diag_values(P_backup)
        try:
            if hasattr(track, 'get_average_velocity') and hasattr(track, 'get_smooth_velocity_trend'):
                smooth_vel = track.get_average_velocity(window=3)
                trend = track.get_smooth_velocity_trend(window=3)
                predicted_vel = smooth_vel + trend * dt * 0.1
            else:
                predicted_vel = get_velocity(track)
            track.kf_3d.kf.x[7:10] = predicted_vel.reshape((3, 1))
            track.kf_3d.kf.x[:3] = track.kf_3d.kf.x[:3] - track.kf_3d.kf.x[7:10] * float(dt)
            track.update_3d(detection)
            diag_after_update = _safe_diag_values(track.kf_3d.kf.P.copy())
            for _ in range(dt):
                track.kf_3d.kf.predict()
            diag_after_fast_forward = _safe_diag_values(track.kf_3d.kf.P.copy())
            _append_covariance_diag_log(
                virtual_update_config,
                track,
                detection_frame_id,
                current_frame,
                dt,
                decay,
                diag_before,
                diag_after_update,
                diag_after_fast_forward,
            )
            _append_l25_hit_event_log(
                virtual_update_config,
                track,
                detection_frame_id,
                current_frame,
                dt,
                decay,
                time_since_update_before,
            )
            track.fusion_time_update += 1
            track.last_backtrack_dt = dt
            track.last_decay_factor = decay
            track.last_l25_recovery_frame = current_frame
            try:
                if hasattr(track, 'reset_rassa'):
                    track.reset_rassa()
            except Exception:
                pass
            updated_tracks.append(track)
            if verbose:
                print(f"[多帧更新完成] 轨迹{track.track_id_3d}: Δt={dt}, 衰减因子={decay:.4f}")
        except Exception:
            track.kf_3d.kf.x = x_backup
            track.kf_3d.kf.P = P_backup
    
    return updated_tracks
