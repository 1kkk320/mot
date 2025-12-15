import numpy as np
import math
import time
from tracking import kalman_filter_2d
from tracking.cost_function import (iou_batch, get_velocity, compute_velocity_similarity, 
                                     estimate_detection_velocity, compute_velocity_trend_similarity,
                                     compute_smooth_velocity_similarity, compute_adaptive_weight_linear)
from tracking.matching import associate_detections_to_trackers_fusion, associate_2D_to_3D_tracking, linear_assignment,associate_detections_to_tracks
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D
from trackers.ocsort_embedding.embedding import EmbeddingComputer
from tracking.matching import compute_aw_new_metric
from tracking.angle_feature import AngleFeatureConfig, compute_angle_similarity_matrix
from tracking.multi_frame_backtrack import (MultiFrameBacktrackConfig, 
                                           multi_frame_backtrack_association,
                                           process_multi_frame_matches)


class Tracker:
    def __init__(self, max_age, n_init, embeddiong_off=False, aw_off=False, grid_off=False,app_off=True,**kwargs):
        self.max_age = max_age
        self.n_init = n_init
        self.tracks_3d = []
        self.tracks_2d = []
        self.track_id_3d = 0   # The id of 3D track is represented by an even number.3d轨迹id由偶数表示
        self.track_id_2d = 1   # The id of 2D track is represented by an odd number.2d轨迹ID为奇数
        self.unmatch_tracks_3d = []
        self.kf_2d = kalman_filter_2d.KalmanFilter()
        self.embedding_off = embeddiong_off
        self.aw_off = aw_off
        self.det_thresh = 0.2
        self.alpha_fixed_emb = 0.9
        self.grid_off = grid_off
        self.app_off = app_off
        self.mot_off = False
        self.embedder = EmbeddingComputer(grid_off=self.grid_off)
        self.appearance_weight_level1 = kwargs.get('appearance_weight_level1', 0.15)
        
        # ========== 速度自适应回溯关联参数 ==========
        self.velocity_backtrack_enabled = True   # 启用速度回溯（创新开启）
        self.velocity_threshold = 0.6            # 速度相似度阈值
        self.velocity_weight = 0.5               # 默认速度权重 (0.5表示速度和位置各占50%)
        self.adaptive_weight = True              # 启用自适应速度权重 (方案B)
        self.detection_history = {}              # 历史检测缓存 {frame_id: detections}
        self.current_frame = 0                   # 当前帧计数
        self.velocity_weight_vmax = 10.0
        self.adaptive_threshold_low = 0.55
        self.adaptive_threshold_mid = 0.60
        self.adaptive_threshold_high = 0.70
        
        # ========== 方案3: 多帧回溯参数 ==========
        self.use_velocity_trend = True           # 启用速度趋势（创新开启）
        self.use_smooth_velocity = True          # 启用速度平滑（创新开启）
        self.velocity_smooth_window = 3          # 速度平滑窗口大小
        self.trend_weight = 0.3                  # 趋势权重 (0.3表示30%趋势 + 70%当前速度)
        
        # ========== 多帧关联配置 ==========
        self.multi_frame_config = MultiFrameBacktrackConfig()
        self.multi_frame_config.enable_multi_frame_backtrack = False  # ❌ 禁用多帧关联
        self.multi_frame_config.min_backtrack_age = 3
        self.multi_frame_config.max_backtrack_age = 15
        self.multi_frame_config.lambda_decay = 0.05
        self.multi_frame_config.cost_threshold = -0.3
        self.multi_frame_config.verbose = False
        self.enable_angle_in_level1 = True       # 启用角度特征（创新开启）
        self.enable_angle_gate_in_backtrack = False
        self.angle_level1_weight = 0.25
        self.angle_level1_method = 'gaussian'
        self.angle_level1_sigma = 0.35
        self.angle_level1_gate_threshold = math.radians(45)
        self.angle_backtrack_method = 'gaussian'
        self.angle_backtrack_sigma = 0.30
        self.angle_backtrack_gate_threshold = math.radians(60)
        self.angle_config = AngleFeatureConfig()
        self.angle_config.enable_angle_feature = self.enable_angle_in_level1
        self.angle_config.angle_cost_method = self.angle_level1_method
        self.angle_config.angle_cost_sigma = self.angle_level1_sigma
        self.angle_config.enable_angle_gate = True   # 启用角度门控（创新开启）
        self.angle_config.angle_gate_threshold = self.angle_level1_gate_threshold
        self.angle_config.angle_weight = self.angle_level1_weight
        self.angle_config.verbose = False
        self.backtrack_angle_gate_verbose = True
        self.t_L1 = 0.0
        self.t_L15 = 0.0
        self.t_L2 = 0.0
        self.t_L3 = 0.0
        self.t_L4 = 0.0

    def predict_3d(self):
        # print(self.tracks_3d)
        for track in self.tracks_3d:
            # print(track)
            track.predict_3d(track.kf_3d)

    def predict_2d(self):
        # print(self.tracks_2d)
        for track in self.tracks_2d:
            # print(track)
            track.predict_2d(self.kf_2d)

    def update(self, detection_3D_fusion, detection_3D_only, detection_3Dto2D_only, detection_2D_only, calib_file, img, detection_2D_only_conf, detection_3D_fusion_conf, iou_threshold):

        # 恢复：初始化占位嵌入，随后在未关闭embedding时再计算真实特征
        dets_3D_fusion_embs = np.ones((len(detection_3D_fusion), 1))
        dets_3D_only_embs = np.ones((len(detection_3D_only), 1))
        dets_2D_only_embs = np.ones((len(detection_2D_only), 1))

        det_3D_fusion_bboxs = [det_3d_f.additional_info[2:6] for det_3d_f in detection_3D_fusion]
        dets_fusion_alpha = None
        dets_2d_only_alpha = None
        # 恢复：一级使用外观（若未关闭embedding）
        use_app_L1 = True
        if use_app_L1 and not self.embedding_off and len(detection_3D_fusion) > 0:
            dets_3D_fusion_embs = self.embedder.compute_embedding(img, det_3D_fusion_bboxs)
        if len(detection_3D_fusion_conf) != 0:
            trust_fusion = np.asarray([(i - self.det_thresh) / (1 - self.det_thresh) for i in detection_3D_fusion_conf])
        else:
            trust_fusion = np.asarray(list())
        if len(detection_2D_only_conf) != 0:
            trust_2D_only = np.asarray([(i - self.det_thresh) / (1 - self.det_thresh) for i in detection_2D_only_conf])
        else:
            trust_2D_only = np.asarray(list())
        af = self.alpha_fixed_emb
        if len(trust_fusion) != 0:
            dets_fusion_alpha = af + (1 - af) * (1 - trust_fusion)
        if len(trust_2D_only) != 0:
            dets_2d_only_alpha = af + (1 - af) * (1 - trust_2D_only)

        # 更新帧计数 (在关联之前)
        self.current_frame += 1

        # 1st Level of Association
        t0 = time.time()
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_detections_to_trackers_fusion(
            detection_3D_fusion, self.tracks_3d, self.aw_off, self.grid_off, self.mot_off, iou_threshold,
            det_embs=dets_3D_fusion_embs, det_app=False, angle_config=self.angle_config,
            enable_angle=self.enable_angle_in_level1, appearance_weight=self.appearance_weight_level1)
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(detection_3D_fusion[detection_idx])
            if use_app_L1 and dets_3D_fusion_embs.shape[0] > detection_idx:
                if dets_fusion_alpha is not None and dets_fusion_alpha.shape[0] > detection_idx:
                    self.tracks_3d[track_idx].update_emb(dets_3D_fusion_embs[detection_idx], alpha=dets_fusion_alpha[detection_idx])
                else:
                    self.tracks_3d[track_idx].update_emb(dets_3D_fusion_embs[detection_idx], alpha=self.alpha_fixed_emb)
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        self.t_L1 += time.time() - t0
        # ========== Level 1.5: 速度自适应回溯关联 ==========
        # 在处理未匹配之前,先尝试速度回溯
        if self.velocity_backtrack_enabled and \
           len(unmatched_dets_fusion_idx) > 0 and \
           len(unmatched_trks_fusion_idx) > 0:
            t1 = time.time()
            print(f"[速度回溯] 未匹配检测: {len(unmatched_dets_fusion_idx)}, "
                  f"未匹配轨迹: {len(unmatched_trks_fusion_idx)}")
            
            # 提取未匹配的检测和轨迹
            unmatched_dets = [detection_3D_fusion[i] for i in unmatched_dets_fusion_idx]
            unmatched_trks = [self.tracks_3d[i] for i in unmatched_trks_fusion_idx]
            
            # 速度回溯关联
            velocity_matched, velocity_unmatched_dets, velocity_unmatched_trks = \
                self._velocity_backtrack_association(
                    unmatched_dets, 
                    unmatched_trks,
                    dets_3D_fusion_embs,
                    unmatched_dets_fusion_idx
                )
            
            # 更新匹配成功的轨迹
            for det_idx, trk_idx in velocity_matched:
                original_det_idx = unmatched_dets_fusion_idx[det_idx]
                original_trk_idx = unmatched_trks_fusion_idx[trk_idx]
                
                self.tracks_3d[original_trk_idx].update_3d(
                    detection_3D_fusion[original_det_idx]
                )
                if not self.app_off:
                    self.tracks_3d[original_trk_idx].update_emb(
                        dets_3D_fusion_embs[original_det_idx]
                    )
                self.tracks_3d[original_trk_idx].state = 2
                self.tracks_3d[original_trk_idx].fusion_time_update = 0
                
                print(f"[速度回溯] ✅ 成功匹配: 检测{original_det_idx} ↔ 轨迹{original_trk_idx}")
            
            # 更新未匹配列表 (只保留真正未匹配的)
            unmatched_dets_fusion_idx = [
                unmatched_dets_fusion_idx[i] for i in velocity_unmatched_dets
            ]
            unmatched_trks_fusion_idx = [
                unmatched_trks_fusion_idx[i] for i in velocity_unmatched_trks
            ]
            
            if len(velocity_matched) > 0:
                print(f"[速度回溯] 📊 本帧匹配成功: {len(velocity_matched)}对")
            self.t_L15 += time.time() - t1
                        
        # ========== 多帧关联 (Level 2.5) ==========
        # 在速度回溯关联后，对仍未匹配的轨迹进行多帧历史关联
        if self.multi_frame_config.enable_multi_frame_backtrack and \
           len(unmatched_dets_fusion_idx) > 0 and \
           len(unmatched_trks_fusion_idx) > 0:
            
            print(f"[多帧关联] 尝试多帧回溯: 未匹配检测{len(unmatched_dets_fusion_idx)}, "
                  f"未匹配轨迹{len(unmatched_trks_fusion_idx)}")
            
            # 提取未匹配的检测和轨迹
            unmatched_dets_mf = [detection_3D_fusion[i] for i in unmatched_dets_fusion_idx]
            unmatched_trks_mf = [self.tracks_3d[i] for i in unmatched_trks_fusion_idx]
            
            # 执行多帧回溯关联
            multi_frame_matches = multi_frame_backtrack_association(
                unmatched_trks_mf,
                self.detection_history,
                self.current_frame,
                self.multi_frame_config
            )
            
            # 处理多帧匹配结果
            if len(multi_frame_matches) > 0:
                updated_tracks = process_multi_frame_matches(
                    multi_frame_matches,
                    verbose=self.multi_frame_config.verbose
                )
                
                # 更新嵌入特征
                for track in updated_tracks:
                    if not self.app_off:
                        # 查找对应的检测嵌入
                        for i, det in enumerate(detection_3D_fusion):
                            if np.allclose(det.bbox[:3], track.pose[:3], atol=0.1):
                                track.update_emb(dets_3D_fusion_embs[i])
                                break
                    
                    # 从未匹配列表中移除已匹配的轨迹
                    # 在self.tracks_3d中找到该轨迹的索引
                    for idx, t in enumerate(self.tracks_3d):
                        if t.track_id_3d == track.track_id_3d:
                            if idx in unmatched_trks_fusion_idx:
                                unmatched_trks_fusion_idx.remove(idx)
                            break
                
                print(f"[多帧关联] ✅ 成功匹配: {len(multi_frame_matches)}对")
            else:
                print(f"[多帧关联] ❌ 未找到匹配")  
        
        # 处理最终未匹配的轨迹和检测
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self._initiate_track_3d(detection_3D_fusion[detection_idx], dets_3D_fusion_embs[detection_idx])

        #  2nd Level of Association
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        t2 = time.time()
        matched_only_idx, unmatched_dets_only_idx, _ = associate_detections_to_trackers_fusion(
            detection_3D_only, self.unmatch_tracks_3d1, self.aw_off, self.grid_off, self.mot_off, iou_threshold,
            det_embs=dets_3D_only_embs, det_app=self.app_off)
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(detection_3D_only[detection_idx])
                    if not self.app_off:
                        t.update_emb(dets_3D_only_embs[detection_idx])
                    index_to_delete.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        for detection_idx in unmatched_dets_only_idx:
            self._initiate_track_3d(detection_3D_only[detection_idx], dets_3D_only_embs[detection_idx])
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1 ]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2
        self.t_L2 += time.time() - t2

        # 3rd Level of Association
        t3 = time.time()
        matched, unmatch_trks, unmatch_dets = associate_detections_to_tracks(self.tracks_2d, detection_2D_only, iou_threshold, self.aw_off,self.grid_off,self.mot_off, det_embs=dets_2D_only_embs, det_app = self.app_off)
        for track_idx, detection_idx in matched:
            self.tracks_2d[track_idx].update_2d(self.kf_2d, detection_2D_only[detection_idx])
            if not self.app_off:
                self.tracks_2d[track_idx].update_emb(dets_2D_only_embs[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self._initiate_track_2d(detection_2D_only[detection_idx], dets_2D_only_embs[detection_idx])
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]
        self.t_L3 += time.time() - t3

        #  4th Level of Association
        t4 = time.time()
        matched_track_2d, unmatch_tracks_2d = associate_2D_to_3D_tracking(self.tracks_2d, self.unmatch_tracks_3d, calib_file, iou_threshold)
        index_to_delete2 = []
        for track_idx_2d, track_idx_3d in matched_track_2d:
            for i in range(len(self.tracks_3d)):
                if self.tracks_3d[i].track_id_3d == self.unmatch_tracks_3d[track_idx_3d].track_id_3d:
                    self.tracks_3d[i].age = self.tracks_2d[track_idx_2d].age + 1
                    if self.tracks_3d[i].track_id_3d % 2 ==0:
                        new_id = self.tracks_2d[track_idx_2d].track_id_2d
                        # ========== 修复: 检查ID唯一性 ==========
                        existing_ids = [t.track_id_3d for t in self.tracks_3d if t != self.tracks_3d[i]]
                        if new_id not in existing_ids:
                            # print(self.tracks_3d[i].track_id_3d,self.tracks_2d[track_idx_2d].track_id_2d)
                            self.tracks_3d[i].track_id_3d = new_id
                            # print("recite:",self.tracks_3d[i].track_id_3d)
                        else:
                            print(f"⚠️ 跳过ID修改: {new_id} 已存在于tracks_3d中")
                        # ========================================
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            index_to_delete2.append(track_idx_2d)
        self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]
        self.t_L4 += time.time() - t4
        
        # ========== DEBUG: 检查重复ID ==========
        track_ids = [t.track_id_3d for t in self.tracks_3d if t.is_confirmed()]
        if len(track_ids) != len(set(track_ids)):
            from collections import Counter
            duplicates = [id for id, count in Counter(track_ids).items() if count > 1]
            print(f"❌ 警告：发现重复的轨迹ID: {duplicates}")
            print(f"   所有ID: {track_ids}")
            print(f"   帧号: {self.current_frame}")
        # ========================================
        
        # 更新检测历史 (帧计数已在函数开始时更新)
        self._update_detection_history(detection_3D_fusion)

    def _velocity_backtrack_association(self, detections, tracks, det_embs, det_indices):
        """
        多层级回溯关联 - 根据轨迹年龄应用不同的参数
        
        Args:
            detections: 未匹配的检测列表
            tracks: 未匹配的轨迹列表
            det_embs: 检测的嵌入特征
            det_indices: 检测在原列表中的索引
        
        Returns:
            matched: 匹配的索引对 [(det_idx, trk_idx), ...]
            unmatched_dets: 未匹配的检测索引
            unmatched_trks: 未匹配的轨迹索引
        """
        if len(detections) == 0 or len(tracks) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # ========== 场景自适应参数配置 ==========
        # 识别场景类型并获取自适应参数
        scene_type = self._identify_scene_type(detections, tracks)
        adaptive_config = self._get_adaptive_backtrack_config(scene_type)
        
        # 如果场景不适合回溯，直接返回
        if not adaptive_config['enable_backtrack']:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # 使用自适应参数
        velocity_weight = adaptive_config['velocity_weight']
        position_weight = adaptive_config['position_weight']
        velocity_threshold = adaptive_config['velocity_threshold']
        max_backtrack_age = adaptive_config['max_backtrack_age']
        
        # 1. 计算速度相似度矩阵
        velocity_matrix = np.zeros((len(detections), len(tracks)))
        
        if self.use_velocity_trend:
            trend_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det in enumerate(detections):
            det_vel = estimate_detection_velocity(det, self.detection_history, self.current_frame)
            
            for t, trk in enumerate(tracks):
                if self.use_smooth_velocity:
                    velocity_matrix[d, t] = compute_smooth_velocity_similarity(
                        trk, det_vel, window=self.velocity_smooth_window
                    )
                else:
                    trk_vel = get_velocity(trk)
                    velocity_matrix[d, t] = compute_velocity_similarity(trk_vel, det_vel)
                
                if self.use_velocity_trend:
                    trend_matrix[d, t] = compute_velocity_trend_similarity(
                        trk, det_vel, use_smooth=True
                    )
        
        if self.use_velocity_trend:
            velocity_matrix = (
                (1 - self.trend_weight) * velocity_matrix + 
                self.trend_weight * trend_matrix
            )
        
        # 2. 计算位置预测相似度 (基于速度预测)
        position_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                if self.use_velocity_trend and hasattr(trk, 'get_velocity_trend'):
                    if self.use_smooth_velocity:
                        trk_vel = trk.get_average_velocity(window=self.velocity_smooth_window)
                    else:
                        trk_vel = get_velocity(trk)
                    
                    if hasattr(trk, 'get_smooth_velocity_trend'):
                        trk_trend = trk.get_smooth_velocity_trend(window=self.velocity_smooth_window)
                    else:
                        trk_trend = trk.get_velocity_trend()
                    
                    dt = 0.1
                    acceleration = self._compute_acceleration(trk)
                    predicted_pos = (
                        trk.pose[:3] + 
                        trk_vel[:3] * dt + 
                        0.5 * acceleration[:3] * dt**2
                    )
                else:
                    trk_vel = get_velocity(trk)
                    predicted_pos = trk.pose[:3] + trk_vel[:3] * 0.1
                
                dist = np.linalg.norm(det.bbox[:3] - predicted_pos)
                position_matrix[d, t] = np.exp(-dist / 5.0)
        
        
        combined_matrix = np.zeros((len(detections), len(tracks)))
        track_weights = []
        for trk in tracks:
            trk_vel = get_velocity(trk)
            w_vel_t, w_pos_t = compute_adaptive_weight_linear(trk_vel, self.velocity_weight_vmax)
            track_weights.append((w_vel_t, w_pos_t))
        for d in range(len(detections)):
            for t in range(len(tracks)):
                w_vel_t, w_pos_t = track_weights[t]
                combined_matrix[d, t] = (
                    w_vel_t * velocity_matrix[d, t] +
                    w_pos_t * position_matrix[d, t]
                )
        
        if self.enable_angle_gate_in_backtrack:
            track_angles = np.array([trk.pose[6] if len(trk.pose) >= 7 else 0.0 for trk in tracks])
            det_angles = np.array([det.bbox[6] if len(det.bbox) >= 7 else 0.0 for det in detections])
            _, gate_mask = compute_angle_similarity_matrix(
                track_angles,
                det_angles,
                method=self.angle_backtrack_method,
                sigma=self.angle_backtrack_sigma,
                gate_threshold=self.angle_backtrack_gate_threshold
            )
            gate_mask_T = gate_mask.T
            if getattr(self, 'backtrack_angle_gate_verbose', False):
                n_total_pairs = int(gate_mask.size)
                n_rejected = int((~gate_mask).sum())
                ratio = (n_rejected / n_total_pairs * 100.0) if n_total_pairs > 0 else 0.0
                print(f"[回溯角度门控] 拒绝 {n_rejected}/{n_total_pairs} 对 ({ratio:.2f}%)")
            combined_matrix[~gate_mask_T] = -1e9
        
        # 4. 匈牙利算法求解
        cost_matrix = -combined_matrix
        matched_indices = linear_assignment(cost_matrix)
        
        # 5. 过滤低相似度匹配
        matches = []
        unmatched_dets = []
        unmatched_trks = []
        
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)
        
        for t in range(len(tracks)):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)
        
        for d, t in matched_indices:
            if combined_matrix[d, t] < velocity_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matches.append([d, t])
        
        return matches, unmatched_dets, unmatched_trks
    
    def _get_adaptive_velocity_weight(self, track):
        """
        根据轨迹速度大小动态调整速度权重 (方案B: 轻量级优化)
        
        Args:
            track: 轨迹对象
        
        Returns:
            weight: 自适应速度权重
        """
        if not self.adaptive_weight:
            return self.velocity_weight
        
        # 获取轨迹速度
        velocity = get_velocity(track)
        speed = np.linalg.norm(velocity)
        
        # 根据速度大小调整权重
        if speed > 15.0:  # 高速场景 (>54 km/h)
            weight = 0.7  # 提高速度权重，速度信息更可靠
            # print(f"[自适应权重] 高速轨迹 {track.id}: {speed:.2f} m/s, 权重={weight}")
        elif speed < 3.0:  # 低速场景 (<10.8 km/h)
            weight = 0.3  # 降低速度权重，位置信息更重要
            # print(f"[自适应权重] 低速轨迹 {track.id}: {speed:.2f} m/s, 权重={weight}")
        else:  # 中速场景 (3-15 m/s)
            weight = 0.5  # 默认权重
        
        return weight
    
    def _update_detection_history(self, detections):
        """
        更新检测历史 (用于速度估计)
        """
        self.detection_history[self.current_frame] = detections
        
        # 只保留最近5帧
        if len(self.detection_history) > 5:
            oldest_frame = min(self.detection_history.keys())
            del self.detection_history[oldest_frame]

    def _initiate_track_3d(self, detection,emb=None):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.n_init, self.max_age,self.additional_info,emb))
        self.track_id_3d += 2

    def _initiate_track_2d(self, detection,emb):
        mean, covariance = self.kf_2d.initiate(detection.to_xyah())
        self.tracks_2d.append(Track_2D(mean, covariance, self.track_id_2d, self.n_init, self.max_age,emb))
        self.track_id_2d += 2
    
    def _identify_scene_type(self, detections, tracks):
        """
        识别场景类型 (低速/中速/高速)
        
        Returns:
            scene_type: 'low_speed_stable', 'medium_speed_mixed', 'high_speed_unstable'
        """
        if len(tracks) == 0:
            return 'unknown'
        
        # 计算平均速度和速度波动
        velocities = []
        for trk in tracks:
            vel = get_velocity(trk)
            speed = np.linalg.norm(vel)
            velocities.append(speed)
        
        avg_speed = np.mean(velocities)
        speed_std = np.std(velocities) if len(velocities) > 1 else 0
        
        # 场景分类
        if avg_speed < 5.0 and speed_std < 1.0:
            return 'low_speed_stable'      # 低速稳定
        elif avg_speed < 15.0 and speed_std < 3.0:
            return 'medium_speed_mixed'    # 中速混合
        else:
            return 'high_speed_unstable'   # 高速不稳定
    
    def _get_adaptive_backtrack_config(self, scene_type):
        """
        根据场景类型返回自适应的回溯参数
        
        Args:
            scene_type: 场景类型
        
        Returns:
            config: 配置字典
        """
        if scene_type == 'low_speed_stable':
            return {
                'velocity_weight': 0.3,
                'position_weight': 0.7,
                'velocity_threshold': self.adaptive_threshold_low,
                'max_backtrack_age': 30,
                'enable_backtrack': True
            }
        elif scene_type == 'medium_speed_mixed':
            return {
                'velocity_weight': 0.4,
                'position_weight': 0.6,
                'velocity_threshold': self.adaptive_threshold_mid,
                'max_backtrack_age': 20,
                'enable_backtrack': True
            }
        else:  # high_speed_unstable
            return {
                'velocity_weight': 0.5,
                'position_weight': 0.5,
                'velocity_threshold': self.adaptive_threshold_high,
                'max_backtrack_age': 10,
                'enable_backtrack': True
            }
    
    def _compute_acceleration(self, track):
        """
        计算轨迹的加速度
        使用速度历史计算: a = (v_new - v_old) / 框帧差
        
        Args:
            track: 轨迹对象
        
        Returns:
            acceleration: 3D 加速度向量
        """
        if not hasattr(track, 'velocity_history') or len(track.velocity_history) < 2:
            return np.zeros(3)
        
        # 获取最近的两个速度
        recent_vel = track.velocity_history[-1][1]  # 最新速度
        prev_vel = track.velocity_history[-2][1]    # 前一个速度
        
        # 计算框帧差
        recent_frame = track.velocity_history[-1][0]
        prev_frame = track.velocity_history[-2][0]
        frame_diff = max(recent_frame - prev_frame, 1)  # 不能为0
        
        # 计算加速度
        acceleration = (recent_vel - prev_vel) / frame_diff
        
        return acceleration