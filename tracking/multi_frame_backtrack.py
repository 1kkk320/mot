"""
多帧关联模块 - 使用衰减时间因子
用于恢复超过3帧未关联的轨迹
"""

import numpy as np
from tracking.cost_function import get_velocity, compute_adaptive_weight_linear, estimate_detection_velocity, compute_velocity_similarity as compute_velocity_similarity_vec
from tracking.matching import linear_assignment


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
        self.uncertainty_norm = 3.0
        # 使用全局最优（线性分配）代替贪心
        self.use_global_assignment = True
        # 非线性回推开关（考虑加速度）
        self.use_nonlinear_backtrack = False    # 基础开关
        # 加速度门控：只在加速度显著时使用非线性
        self.use_acceleration_gate = True       # ✅ 启用加速度门控
        self.acceleration_threshold = 1.5       # 加速度阈值 (m/s²)，超过此值才用非线性
        # 软门控参数（平滑过渡）
        self.use_soft_gate = True               # ✅ 启用软门控（替代硬门控）
        self.soft_gate_sharpness = 2.0          # 陡峭度（越大越接近硬门控）


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


def soft_acceleration_gate(a_norm, threshold=1.5, sharpness=2.0):
    """
    软门控函数：使用 Sigmoid 函数实现平滑过渡
    
    Args:
        a_norm: 加速度的模（标量）
        threshold: 中心阈值（加速度显著性的分界点）
        sharpness: 陡峭度（控制过渡的平滑程度）
                   - 越大越接近硬门控（陡峭）
                   - 越小越平滑（缓慢过渡）
    
    Returns:
        weight: 非线性权重 [0, 1]
                0 = 完全线性
                1 = 完全非线性
    
    示例（threshold=1.5, sharpness=2.0）：
        a_norm=0.5  → weight≈0.05  (几乎线性)
        a_norm=1.0  → weight≈0.27  (偏线性)
        a_norm=1.5  → weight=0.50  (中间)
        a_norm=2.0  → weight≈0.73  (偏非线性)
        a_norm=2.5  → weight≈0.92  (几乎非线性)
    """
    weight = 1.0 / (1.0 + np.exp(-sharpness * (a_norm - threshold)))
    return float(weight)


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
        
        # 加速度门控参数
        use_acceleration_gate = getattr(config, 'use_acceleration_gate', False) if config else False
        acceleration_threshold = getattr(config, 'acceleration_threshold', 1.5) if config else 1.5
        use_soft_gate = getattr(config, 'use_soft_gate', False) if config else False
        soft_gate_sharpness = getattr(config, 'soft_gate_sharpness', 2.0) if config else 2.0
        
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
                    # 计算该时刻的加速度（简单差分）
                    if i > 0:
                        prev_frame, prev_vel = track.velocity_history[i-1]
                        dt = max(frame_id - prev_frame, 1)
                        a_at_target = (vel - prev_vel) / dt
                        if verbose and np.random.rand() < 0.01:
                            print(f"[简单差分] |a_simple|={float(np.linalg.norm(a_at_target)):.3f}")
                    break
            
            # 如果找到了历史速度，使用历史速度回推
            if v_at_target is not None:
                dt = float(time_diff)
                v_current = x[7:10].reshape(3)
                
                # 加速度门控判断
                nonlinear_weight = 0.0  # 默认完全线性
                
                if use_acceleration_gate and a_at_target is not None:
                    a_norm = float(np.linalg.norm(a_at_target))
                    
                    if use_soft_gate:
                        # 软门控：平滑过渡
                        nonlinear_weight = soft_acceleration_gate(
                            a_norm, 
                            threshold=acceleration_threshold,
                            sharpness=soft_gate_sharpness
                        )
                        
                        if verbose and np.random.rand() < 0.01:
                            print(f"[软门控] |a|={a_norm:.3f}, threshold={acceleration_threshold:.1f}, "
                                  f"weight={nonlinear_weight:.3f} ({'线性' if nonlinear_weight < 0.3 else '混合' if nonlinear_weight < 0.7 else '非线性'})")
                    else:
                        # 硬门控：二值决策
                        if a_norm >= acceleration_threshold:
                            nonlinear_weight = 1.0
                            if verbose and np.random.rand() < 0.01:
                                print(f"[硬门控] |a|={a_norm:.3f} >= {acceleration_threshold:.1f}, 使用非线性")
                        else:
                            nonlinear_weight = 0.0
                            if verbose and np.random.rand() < 0.01:
                                print(f"[硬门控] |a|={a_norm:.3f} < {acceleration_threshold:.1f}, 使用线性")
                
                if verbose and np.random.rand() < 0.01:
                    v_norm_target = float(np.linalg.norm(v_at_target))
                    v_norm_current = float(np.linalg.norm(v_current))
                    print(f"[历史速度回推] Δt={dt:.0f}, v_target={v_norm_target:.2f}, "
                          f"v_current={v_norm_current:.2f}, 差异={abs(v_norm_target-v_norm_current):.2f}")
                
                # 使用软门控权重混合线性和非线性
                if a_at_target is not None and nonlinear_weight > 0:
                    # 混合：weight * 非线性 + (1-weight) * 线性
                    # 非线性项：-0.5 * a * dt²
                    # 线性项：0
                    pos = current_pos - v_at_target * dt - nonlinear_weight * 0.5 * a_at_target * (dt ** 2)
                else:
                    # 完全线性（没有加速度或权重为0）
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
    frames = []
    for fid in detection_buffer.keys():
        dt = current_frame - fid
        if dt > 0 and dt <= getattr(config, 'last_k_frames', 5):
            frames.append((dt, fid))
    frames.sort(key=lambda x: x[0])  # t-1, t-2, ...

    for dt, fid in frames:
        detections = detection_buffer.get(fid, [])
        decay = compute_decay_factor(dt, config.lambda_decay)
        per_frame = []
        for det in detections:
            # 相似度项（不使用角/角速度）
            # 方案A：将轨迹回推到历史帧，与历史检测计算IoU
            use_nonlinear = getattr(config, 'use_nonlinear_backtrack', True)
            rollback_pose = get_pose_at_past_frame(track, dt, use_nonlinear=use_nonlinear, config=config)
            iou = compute_iou_3d(rollback_pose, det.bbox)
            if iou <= 1e-6:
                continue
            app_sim = compute_appearance_similarity(track, det)
            if app_sim < getattr(config, 'appearance_hard_gate', 0.6):
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
            # 速度/位置残差分配 + 协方差不确定性抑制速度
            vmax = getattr(config, 'vmax_for_adaptive_weight', 10.0)
            w_vel_t, w_pos_t = compute_adaptive_weight_linear(get_velocity(track), v_max=vmax)
            # 估计不确定性: 取x/z方差
            uncertainty = 0.0
            try:
                P = track.kf_3d.kf.P
                sx = float(np.sqrt(np.abs(P[0, 0]))) if P.shape[0] > 0 else 0.0
                sz = float(np.sqrt(np.abs(P[2, 2]))) if P.shape[0] > 2 else 0.0
                u_norm = getattr(config, 'uncertainty_norm', 3.0)
                uncertainty = max(0.0, min(1.0, (sx + sz) / max(u_norm, 1e-6)))
            except Exception:
                uncertainty = 0.0
            w_vel = residual * w_vel_t * (1.0 - uncertainty)
            # 速度权重下限
            w_vel = max(0.1, w_vel)
            w_vel = min(w_vel, residual)
            w_iou = max(0.0, residual - w_vel)
            # 融合相似度并应用时间衰减
            combined_sim = w_iou * iou + w_vel * vel_sim + w_app * app_sim
            decayed_sim = combined_sim * decay
            cost = -decayed_sim
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

    # 路径1：全局最优（线性分配）
    if getattr(config, 'use_global_assignment', False):
        # 1) 收集最近K帧的候选检测（按时间就近）
        frames = []
        for fid in detection_buffer.keys():
            dt = current_frame - fid
            if dt > 0 and dt <= getattr(config, 'last_k_frames', 5):
                frames.append((dt, fid))
        frames.sort(key=lambda x: x[0])

        cand_list = []  # [(fid, det, dt, decay)]
        seen = set()
        for dt, fid in frames:
            decay = compute_decay_factor(dt, config.lambda_decay)
            for det in detection_buffer.get(fid, []):
                # 去重：相同对象引用仅保留一次
                det_id = id(det)
                if det_id in seen:
                    continue
                seen.add(det_id)
                cand_list.append((fid, det, dt, decay))

        if len(cand_list) == 0 or len(unmatched_tracks) == 0:
            return matched_pairs

        # 2) 构建代价矩阵（行=候选检测，列=轨迹），以便全局匹配
        INF = 1e6
        C = np.full((len(cand_list), len(unmatched_tracks)), INF, dtype=np.float32)

        for j, (fid, det, dt, decay) in enumerate(cand_list):
            for i, track in enumerate(unmatched_tracks):
                # 触发条件（轨迹年龄）
                if not (config.min_backtrack_age <= track.time_since_update <= config.max_backtrack_age):
                    continue
                # 相似度项
                use_nonlinear = getattr(config, 'use_nonlinear_backtrack', True)
                rollback_pose = get_pose_at_past_frame(track, dt, use_nonlinear=use_nonlinear, config=config)
                iou = compute_iou_3d(rollback_pose, det.bbox)
                if iou <= 1e-6:
                    continue
                app_sim = compute_appearance_similarity(track, det)
                if app_sim < getattr(config, 'appearance_hard_gate', 0.6):
                    continue
                det_vel = estimate_detection_velocity(det, detection_buffer, fid)
                trk_vel = get_velocity(track)
                vel_sim = compute_velocity_similarity_vec(trk_vel, det_vel)
                # 外观权重（自适应上限）
                cap = 0.25 if getattr(track, 'time_since_update', 0) >= 3 else 0.22
                base_app = min(max(getattr(config, 'appearance_weight', 0.2), 0.0), cap)
                reliable = (app_sim >= 0.6)
                w_app = base_app if reliable else min(0.05, base_app)
                residual = max(0.0, 1.0 - w_app)
                # 速度/位置分配 + 不确定性抑制
                vmax = getattr(config, 'vmax_for_adaptive_weight', 10.0)
                w_vel_t, w_pos_t = compute_adaptive_weight_linear(get_velocity(track), v_max=vmax)
                uncertainty = 0.0
                try:
                    P = track.kf_3d.kf.P
                    sx = float(np.sqrt(np.abs(P[0, 0]))) if P.shape[0] > 0 else 0.0
                    sz = float(np.sqrt(np.abs(P[2, 2]))) if P.shape[0] > 2 else 0.0
                    u_norm = getattr(config, 'uncertainty_norm', 3.0)
                    uncertainty = max(0.0, min(1.0, (sx + sz) / max(u_norm, 1e-6)))
                except Exception:
                    uncertainty = 0.0
                w_vel = residual * w_vel_t * (1.0 - uncertainty)
                # 速度权重下限
                w_vel = max(0.1, w_vel)
                w_vel = min(w_vel, residual)
                w_iou = max(0.0, residual - w_vel)
                combined_sim = w_iou * iou + w_vel * vel_sim + w_app * app_sim
                decayed_sim = combined_sim * decay
                cost = -decayed_sim
                # 预门控：若综合相似度太低，则直接屏蔽该对
                if cost >= config.cost_threshold:
                    continue
                C[j, i] = cost
        # 统计：候选对总数、通过预门控数量、decayed_sim 统计
        pairs_total = int(C.shape[0] * C.shape[1])
        if pairs_total > 0:
            mask = (C < INF)
            if mask.any():
                sims = -C[mask]
                smin = float(np.min(sims))
                smean = float(np.mean(sims))
                smax = float(np.max(sims))
                print(f"[L2.5 Stats] pairs={pairs_total}, pass={int(mask.sum())}, decayed_sim(min/mean/max)={smin:.3f}/{smean:.3f}/{smax:.3f}")
            else:
                print(f"[L2.5 Stats] pairs={pairs_total}, pass=0, decayed_sim(min/mean/max)=NA/NA/NA")

        # 3) 线性分配（全局最优）
        assign = linear_assignment(C)
        if assign.size == 0:
            return matched_pairs
        # assign 的格式与 matching.linear_assignment 一致：行索引在 [:,0]，列索引在 [:,1]
        for row_idx, col_idx in assign:
            # 防御：检查是否为有效匹配
            if row_idx < 0 or row_idx >= C.shape[0] or col_idx < 0 or col_idx >= C.shape[1]:
                continue
            pair_cost = float(C[row_idx, col_idx])
            if not np.isfinite(pair_cost) or pair_cost >= config.cost_threshold:
                continue
            fid, det, dt, decay = cand_list[int(row_idx)]
            track = unmatched_tracks[int(col_idx)]
            matched_pairs.append((track, det, fid, dt, decay))
            if config.verbose:
                print(f"[多帧关联-GLOBAL] 轨迹{track.track_id_3d} ⇐ 帧{fid} (Δt={dt}) 代价={pair_cost:.4f}")
        
        # 统计 Δt 分布
        if len(matched_pairs) > 0:
            dt_list = [dt for _, _, _, dt, _ in matched_pairs]
            from collections import Counter
            dt_dist = Counter(dt_list)
            print(f"[L2.5 Δt分布] 总匹配={len(matched_pairs)}, 分布={dict(sorted(dt_dist.items()))}")
        
        return matched_pairs

    # 路径2：保留原贪心（回退）
    for track in unmatched_tracks:
        # 检查是否满足触发条件
        if not (config.min_backtrack_age <= track.time_since_update <= config.max_backtrack_age):
            continue
        # 获取候选并按就近优先顺序检查
        candidates = compute_decay_cost_matrix(
            track, detection_buffer, current_frame, config
        )
        if not candidates:
            continue
        # 顺序扫描，找到第一个过阈值且未被使用的候选
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
        dt = int(max(0, time_diff))
        x_backup = track.kf_3d.kf.x.copy()
        P_backup = track.kf_3d.kf.P.copy()
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
            for _ in range(dt):
                track.kf_3d.kf.predict()
            track.fusion_time_update += 1
            track.last_backtrack_dt = dt
            track.last_decay_factor = decay
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
