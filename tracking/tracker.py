import numpy as np
from tracking import kalman_filter_2d
from tracking.cost_function import (iou_batch, get_velocity, compute_velocity_similarity, 
                                     estimate_detection_velocity, compute_velocity_trend_similarity,
                                     compute_smooth_velocity_similarity)
from tracking.matching import associate_detections_to_trackers_fusion, associate_2D_to_3D_tracking, linear_assignment,associate_detections_to_tracks
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D
from trackers.ocsort_embedding.embedding import EmbeddingComputer
from tracking.matching import compute_aw_new_metric


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
        
        # ========== 速度自适应回溯关联参数 ==========
        self.velocity_backtrack_enabled = True   # 启用速度回溯 ✅
        self.velocity_threshold = 0.6            # 速度相似度阈值
        self.velocity_weight = 0.5               # 默认速度权重 (0.5表示速度和位置各占50%)
        self.adaptive_weight = True              # 启用自适应速度权重 (方案B)
        self.detection_history = {}              # 历史检测缓存 {frame_id: detections}
        self.current_frame = 0                   # 当前帧计数
        
        # ========== 方案3: 多帧回溯参数 ==========
        self.use_velocity_trend = True          # 暂时禁用 (调试重复ID问题) ❌
        self.use_smooth_velocity = True         # 暂时禁用 (调试重复ID问题) ❌
        self.velocity_smooth_window = 3          # 速度平滑窗口大小
        self.trend_weight = 0.3                  # 趋势权重 (0.3表示30%趋势 + 70%当前速度)

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

        # generate embedding
        #初始化
        dets_3D_fusion_embs = np.ones((len(detection_3D_fusion),1))
        dets_3D_only_embs = np.ones((len(detection_3D_only),1))
        dets_2D_only_embs = np.ones((len(detection_2D_only),1))
        det_3D_fusion_bboxs = [det_3d_f.additional_info[2:6] for det_3d_f in detection_3D_fusion]
        det_3D_only_bboxs = [det_3d_f.additional_info[2:6] for det_3d_f in detection_3D_only]
        det_2D_only_bboxs = [det_2d_f.bbox for det_2d_f in detection_2D_only]
        # 得到嵌入特征
        if not self.app_off:
            if not self.embedding_off and dets_3D_fusion_embs.shape[0] != 0:
                dets_3D_fusion_embs = self.embedder.compute_embedding(img,det_3D_fusion_bboxs)
            if not self.embedding_off and dets_3D_only_embs.shape[0] != 0:
                dets_3D_only_embs = self.embedder.compute_embedding(img,det_3D_only_bboxs)
            if not self.embedding_off and dets_2D_only_embs.shape[0] != 0:
                dets_2D_only_embs = self.embedder.compute_embedding(img,det_2D_only_bboxs)
            # 计算嵌入特征的可信度
            if len(detection_3D_fusion_conf) != 0:
                trust_fusion = np.asarray([(i - self.det_thresh) / (1 - self.det_thresh) for i in detection_3D_fusion_conf])
            else:
                trust_fusion = np.asarray(list())
            if len(detection_2D_only_conf) != 0:
                trust_2D_only = np.asarray([(i - self.det_thresh) / (1 - self.det_thresh) for i in detection_2D_only_conf])
            else:
                trust_2D_only = np.asarray(list())
            af = self.alpha_fixed_emb
            # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
            if len(trust_fusion) != 0:
                dets_fusion_alpha = af + (1 - af) * (1 - trust_fusion)
            if len(trust_2D_only) != 0:
                dets_2d_only_alpha = af + (1 - af) * (1 - trust_2D_only)

        # 更新帧计数 (在关联之前)
        self.current_frame += 1

        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_detections_to_trackers_fusion(
            detection_3D_fusion, self.tracks_3d,self.aw_off,self.grid_off,self.mot_off, iou_threshold,det_embs=dets_3D_fusion_embs, det_app = self.app_off)
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(detection_3D_fusion[detection_idx])
            if not self.app_off:
                self.tracks_3d[track_idx].update_emb(dets_3D_fusion_embs[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        # ========== Level 1.5: 速度自适应回溯关联 ==========
        # 在处理未匹配之前,先尝试速度回溯
        if self.velocity_backtrack_enabled and \
           len(unmatched_dets_fusion_idx) > 0 and \
           len(unmatched_trks_fusion_idx) > 0:
            
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
        
        # 处理最终未匹配的轨迹和检测
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()
        for detection_idx in unmatched_dets_fusion_idx:
            self._initiate_track_3d(detection_3D_fusion[detection_idx],dets_3D_fusion_embs[detection_idx])

        #  2nd Level of Association
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        matched_only_idx, unmatched_dets_only_idx, _ = associate_detections_to_trackers_fusion(
            detection_3D_only, self.unmatch_tracks_3d1,self.aw_off,self.grid_off,self.mot_off, iou_threshold, det_embs=dets_3D_only_embs, det_app=self.app_off)
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
            self._initiate_track_3d(detection_3D_only[detection_idx],dets_3D_only_embs[detection_idx])
        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1 ]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # 3rd Level of Association
        matched, unmatch_trks, unmatch_dets = associate_detections_to_tracks(self.tracks_2d, detection_2D_only, iou_threshold, self.aw_off,self.grid_off,self.mot_off, det_embs=dets_2D_only_embs, det_app = self.app_off)
        for track_idx, detection_idx in matched:
            self.tracks_2d[track_idx].update_2d(self.kf_2d, detection_2D_only[detection_idx])
            if not self.app_off:
                self.tracks_2d[track_idx].update_emb(dets_2D_only_embs[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self._initiate_track_2d(detection_2D_only[detection_idx],dets_2D_only_embs[detection_idx])
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]

        #  4th Level of Association
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
        基于速度的回溯关联
        
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
        
        # 1. 计算速度相似度矩阵
        velocity_matrix = np.zeros((len(detections), len(tracks)))
        
        # ========== 方案3: 可选的趋势相似度矩阵 ==========
        if self.use_velocity_trend:
            trend_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det in enumerate(detections):
            # 估计检测速度
            det_vel = estimate_detection_velocity(det, self.detection_history, self.current_frame)
            
            for t, trk in enumerate(tracks):
                # ========== 方案B: 当前速度相似度 ==========
                if self.use_smooth_velocity:
                    # 方案3: 使用平滑速度 (降低噪声)
                    velocity_matrix[d, t] = compute_smooth_velocity_similarity(
                        trk, det_vel, window=self.velocity_smooth_window
                    )
                else:
                    # 方案B: 使用当前速度
                    trk_vel = get_velocity(trk)
                    velocity_matrix[d, t] = compute_velocity_similarity(trk_vel, det_vel)
                
                # ========== 方案3: 趋势相似度 (考虑加速度) ==========
                if self.use_velocity_trend:
                    trend_matrix[d, t] = compute_velocity_trend_similarity(
                        trk, det_vel, use_smooth=True
                    )
        
        # ========== 方案3: 融合当前速度和趋势 ==========
        if self.use_velocity_trend:
            # 融合: (1-w)*当前速度 + w*趋势
            velocity_matrix = (
                (1 - self.trend_weight) * velocity_matrix + 
                self.trend_weight * trend_matrix
            )
        
        # 2. 计算位置预测相似度 (基于速度预测)
        position_matrix = np.zeros((len(detections), len(tracks)))
        weight_matrix = np.zeros((len(detections), len(tracks)))  # 每对的自适应权重
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                # ========== 方案3: 使用趋势预测位置 (改进) ==========
                if self.use_velocity_trend and hasattr(trk, 'get_velocity_trend'):
                    # 使用平滑速度和趋势
                    if self.use_smooth_velocity:
                        trk_vel = trk.get_average_velocity(window=self.velocity_smooth_window)
                    else:
                        trk_vel = get_velocity(trk)
                    
                    # 获取速度趋势
                    if hasattr(trk, 'get_smooth_velocity_trend'):
                        trk_trend = trk.get_smooth_velocity_trend(window=self.velocity_smooth_window)
                    else:
                        trk_trend = trk.get_velocity_trend()
                    
                    # 预测速度 (考虑加速度)
                    predicted_vel = trk_vel + trk_trend * 0.1
                    predicted_pos = trk.pose[:3] + predicted_vel[:3] * 0.1
                else:
                    # ========== 方案B: 简单线性预测 ==========
                    trk_vel = get_velocity(trk)
                    predicted_pos = trk.pose[:3] + trk_vel[:3] * 0.1  # 假设dt=0.1s
                
                # 计算预测位置与检测位置的距离
                dist = np.linalg.norm(det.bbox[:3] - predicted_pos)
                
                # 转换为相似度 (距离越小,相似度越高)
                position_matrix[d, t] = np.exp(-dist / 5.0)  # 5米衰减
                
                # 获取该轨迹的自适应权重
                weight_matrix[d, t] = self._get_adaptive_velocity_weight(trk)
        
        # 3. 融合速度和位置相似度 (使用自适应权重)
        combined_matrix = np.zeros((len(detections), len(tracks)))
        for d in range(len(detections)):
            for t in range(len(tracks)):
                w = weight_matrix[d, t]
                combined_matrix[d, t] = (
                    w * velocity_matrix[d, t] + 
                    (1 - w) * position_matrix[d, t]
                )
        
        # 4. 匈牙利算法求解
        cost_matrix = -combined_matrix  # 转为代价矩阵
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
            if combined_matrix[d, t] < self.velocity_threshold:
                # 相似度太低,拒绝匹配
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