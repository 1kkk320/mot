import numpy as np
import math
import time
import os
from tracking import kalman_filter_2d
from tracking.cost_function import (iou_batch, get_velocity, compute_velocity_similarity, 
                                     estimate_detection_velocity, compute_velocity_trend_similarity,
                                     compute_smooth_velocity_similarity, compute_adaptive_weight_linear)
from tracking.matching import associate_detections_to_trackers_fusion, associate_2D_to_3D_tracking, linear_assignment,associate_detections_to_tracks, compute_rotated_ground_similarity
from tracking.track_2d import Track_2D
from tracking.kalman_fileter_3d import  KalmanBoxTracker
from tracking.track_3d import Track_3D, TrackState
from trackers.ocsort_embedding.embedding import EmbeddingComputer
from tracking.matching import compute_aw_new_metric
from tracking.multi_frame_backtrack import (MultiFrameBacktrackConfig, 
                                           multi_frame_backtrack_association,
                                           process_multi_frame_matches)


class Tracker:
    def __init__(self, max_age, n_init, embeddiong_off=False, aw_off=False, grid_off=False,app_off=False,**kwargs):
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
        self.alpha_fixed_emb = 0.8
        self.grid_off = grid_off
        self.app_off = app_off
        self.mot_off = False
        self.embedder = EmbeddingComputer(grid_off=self.grid_off)
        self.appearance_weight_level1 = kwargs.get('appearance_weight_level1', 0.10)  # 从0.15降到0.10，给角度特征更多权重
        
        # ========== 速度自适应回溯关联参数 ==========
        self.velocity_backtrack_enabled = False  # ❌ 基线：关闭L1.5速度回溯
        self.velocity_threshold = 0.6            # 速度相似度阈值
        self.velocity_weight = 0.5               # 默认速度权重 (0.5表示速度和位置各占50%)
        self.adaptive_weight = True              # 启用自适应速度权重 (方案B)
        self.detection_history = {}              # 历史检测缓存 {frame_id: detections}
        self.current_frame = 0                   # 当前帧计数
        self.velocity_weight_vmax = 12.0
        self.adaptive_threshold_low = 0.45       # 从0.55降到0.45（更宽松）
        self.adaptive_threshold_mid = 0.60
        self.adaptive_threshold_high = 0.75      # 从0.70升到0.75（更宽松）
        
        # ========== 统计计数器 ==========
        self.total_L15_recoveries = 0            # L1.5 总恢复数
        self.total_L25_recoveries = 0            # L2.5 总恢复数
        
        # ========== 方案3: 多帧回溯参数 ==========
        self.use_velocity_trend = True           # 启用速度趋势（创新开启）
        self.use_smooth_velocity = True          # 启用速度平滑（创新开启）
        self.velocity_smooth_window = 3          # 速度平滑窗口大小
        self.trend_weight = 0.3                  # 趋势权重 (0.3表示30%趋势 + 70%当前速度)
        
        # ========== 多帧关联配置 ==========
        self.multi_frame_config = MultiFrameBacktrackConfig()
        self.multi_frame_config.vmax_for_adaptive_weight = 12.0
        self.multi_frame_config.enable_multi_frame_backtrack = False  # ❌ 基线：关闭L2.5多帧回溯
        self.multi_frame_config.min_backtrack_age = 4
        self.multi_frame_config.max_backtrack_age = 15
        self.multi_frame_config.lambda_decay = 0.15
        self.multi_frame_config.cost_threshold = -0.35
        self.multi_frame_config.last_k_frames = 5
        self.multi_frame_config.detection_buffer_size = 5
        self.multi_frame_config.topk_per_frame = 1
        self.multi_frame_config.verbose = False
        self.multi_frame_config.appearance_weight = 0.2
        self.multi_frame_config.appearance_hard_gate = 0.50
        self.multi_frame_config.use_nonlinear_backtrack = False  # 基础开关保持关闭
        self.multi_frame_config.use_acceleration_gate = False   # ❌ 基线：关闭加速度门控
        self.multi_frame_config.acceleration_threshold = 1.5   # 加速度阈值 (m/s²)
        self.heading_ambiguity_rescore_enabled = False
        self.heading_distance_metric = 'symmetric'
        self.heading_ambiguity_margin = 0.08
        self.heading_ambiguity_weight = 0.20
        self.heading_pre_gate_enabled = True
        self.heading_pre_gate_threshold = 0.55
        self.heading_hard_gate_threshold = 0.45
        self.use_rotated_geom_in_l1 = False
        self.use_rotated_geom_in_l2 = False
        self.use_rotated_geom_in_l15 = False
        self.rotated_geom_weight_l15 = 0.20
        self.use_state_heading_in_l1 = False
        self.use_state_heading_in_l2 = False
        self.use_state_heading_in_l15 = False
        self.state_heading_sigma = 0.45
        self.heading_stats = {}
        self.last_t_L1 = 0.0
        self.last_t_L15 = 0.0
        self.last_t_L2 = 0.0
        self.last_t_L3 = 0.0
        self.last_t_L4 = 0.0
        self.last_t_L25 = 0.0
        self._timing_eps = 1e-6
        self.sum_t_L1 = 0.0
        self.cnt_t_L1 = 0
        self.sum_t_L15 = 0.0
        self.cnt_t_L15 = 0
        self.sum_t_L2 = 0.0
        self.cnt_t_L2 = 0
        self.sum_t_L25 = 0.0
        self.cnt_t_L25 = 0
        self.sum_t_L3 = 0.0
        self.cnt_t_L3 = 0
        self.sum_t_L4 = 0.0
        self.cnt_t_L4 = 0

        # ===== 全局开关：多帧回溯（L2.5） =====
        # 可通过环境变量 ENABLE_MULTI_FRAME_BACKTRACK 控制（'1','true','yes','on' 为启用）
        env_flag = os.environ.get('ENABLE_MULTI_FRAME_BACKTRACK', '1').lower()
        self.enable_backtrack_global = env_flag in ('1', 'true', 'yes', 'on')

    def _append_l25_final_hit_log(self, track, initial_track_id):
        cfg = getattr(self, 'multi_frame_config', None)
        if cfg is None or not getattr(cfg, 'enable_final_hit_event_log', False):
            return
        log_path = getattr(cfg, 'final_hit_event_log_path', None)
        if not log_path:
            return
        try:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            seq_id = getattr(cfg, 'current_seq_id', 'unknown')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(
                    '[L2.5 FinalHit] seq={} frame={} initial_track_id={} final_track_id={} '
                    'confirmed={} state={} hits={} tsu={} dt={} decay={:.6f}\n'.format(
                        seq_id,
                        int(getattr(cfg, 'current_data_frame', self.current_frame)),
                        int(initial_track_id),
                        int(getattr(track, 'track_id_3d', -1)),
                        int(track.is_confirmed()),
                        int(getattr(track, 'state', -1)),
                        int(getattr(track, 'hits', -1)),
                        int(getattr(track, 'time_since_update', -1)),
                        int(getattr(track, 'last_backtrack_dt', -1)),
                        float(getattr(track, 'last_decay_factor', 0.0)),
                    )
                )
        except Exception:
            pass

    def _maybe_add_track_memory_feature(self, track, emb, det_score=1.0):
        cfg = getattr(self, 'multi_frame_config', None)
        if cfg is None or not getattr(cfg, 'use_l25_memory_bank_appearance', False):
            return
        if track is None or emb is None:
            return
        try:
            if float(det_score) < float(getattr(cfg, 'memory_bank_min_conf', 0.4)):
                return
        except Exception:
            return
        if not track.is_confirmed():
            return
        try:
            track.add_memory_feature(
                emb,
                max_size=getattr(cfg, 'memory_bank_size', 3)
            )
        except Exception:
            return

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

    def _get_heading_options(self, stage=None):
        return {
            'enabled': self.heading_ambiguity_rescore_enabled,
            'metric': self.heading_distance_metric,
            'ambiguity_margin': self.heading_ambiguity_margin,
            'weight': self.heading_ambiguity_weight,
            'pre_gate_enabled': self.heading_pre_gate_enabled,
            'pre_gate_threshold': self.heading_pre_gate_threshold,
            'hard_gate_threshold': self.heading_hard_gate_threshold,
            'use_state_heading': (
                self.use_state_heading_in_l1 if stage == 'l1'
                else self.use_state_heading_in_l2 if stage == 'l2'
                else False
            ),
            'use_rotated_geom': (
                self.use_rotated_geom_in_l1 if stage == 'l1'
                else self.use_rotated_geom_in_l2 if stage == 'l2'
                else False
            ),
            'state_heading_sigma': self.state_heading_sigma,
            'stats': self.heading_stats,
        }

    def reset_heading_stats(self):
        self.heading_stats = {
            'calls': 0,
            'changed_calls': 0,
            'changed_pairs': 0,
            'pre_gate_suppressed_pairs': 0,
            'pre_gate_changed_matches': 0,
        }

    def _wrap_to_pi(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _compute_state_heading_consistency(self, track, detection, det_vel=None):
        try:
            track_heading = float(track.get_predicted_heading(1.0)) if hasattr(track, 'get_predicted_heading') else float(track.pose[3])
        except Exception:
            return 0.5

        obs_angles = []
        obs_weights = []

        try:
            det_heading = float(detection.bbox[3])
            det_conf = float(getattr(track, 'heading_conf_det', 0.5))
            obs_angles.append(det_heading)
            obs_weights.append(max(det_conf, 1e-3))
        except Exception:
            pass

        if det_vel is None:
            try:
                det_vel = estimate_detection_velocity(detection, self.detection_history, self.current_frame)
            except Exception:
                det_vel = None
        try:
            if det_vel is not None:
                vx = float(det_vel[0])
                vz = float(det_vel[2])
                vel_heading = math.atan2(vz, vx + 1e-6)
                speed = math.sqrt(vx * vx + vz * vz)
                vel_conf = (1.0 - math.exp(-speed / 3.0)) * max(float(getattr(track, 'heading_conf_vel', 0.5)), 1e-3)
                obs_angles.append(vel_heading)
                obs_weights.append(max(vel_conf, 1e-3))
        except Exception:
            pass

        if len(obs_angles) == 0:
            return 0.5

        s = 0.0
        c = 0.0
        for ang, w in zip(obs_angles, obs_weights):
            s += w * math.sin(ang)
            c += w * math.cos(ang)
        fused_obs = math.atan2(s, c) if (abs(s) > 1e-8 or abs(c) > 1e-8) else obs_angles[0]

        delta = self._wrap_to_pi(track_heading - fused_obs)
        sigma = max(float(self.state_heading_sigma), 1e-3)
        return float(math.exp(-0.5 * (delta / sigma) ** 2))

    @property
    def t_L1(self):
        return self.last_t_L1

    @property
    def t_L15(self):
        return self.last_t_L15

    @property
    def t_L2(self):
        return self.last_t_L2

    @property
    def t_L3(self):
        return self.last_t_L3

    @property
    def t_L4(self):
        return self.last_t_L4

    @property
    def t_L25(self):
        return self.last_t_L25

    @property
    def avg_t_L1(self):
        return self.sum_t_L1 / self.cnt_t_L1 if self.cnt_t_L1 > 0 else 0.0

    @property
    def avg_t_L15(self):
        return self.sum_t_L15 / self.cnt_t_L15 if self.cnt_t_L15 > 0 else 0.0

    @property
    def avg_t_L2(self):
        return self.sum_t_L2 / self.cnt_t_L2 if self.cnt_t_L2 > 0 else 0.0

    @property
    def avg_t_L25(self):
        return self.sum_t_L25 / self.cnt_t_L25 if self.cnt_t_L25 > 0 else 0.0

    @property
    def avg_t_L3(self):
        return self.sum_t_L3 / self.cnt_t_L3 if self.cnt_t_L3 > 0 else 0.0

    @property
    def avg_t_L4(self):
        return self.sum_t_L4 / self.cnt_t_L4 if self.cnt_t_L4 > 0 else 0.0

    @property
    def trig_cnt_L1(self):
        return self.cnt_t_L1

    @property
    def trig_cnt_L15(self):
        return self.cnt_t_L15

    @property
    def trig_cnt_L2(self):
        return self.cnt_t_L2

    @property
    def trig_cnt_L25(self):
        return self.cnt_t_L25

    @property
    def trig_cnt_L3(self):
        return self.cnt_t_L3

    @property
    def trig_cnt_L4(self):
        return self.cnt_t_L4

    @property
    def est_one_pass_avg(self):
        return (
            self.avg_t_L1 + self.avg_t_L15 + self.avg_t_L2 +
            self.avg_t_L25 + self.avg_t_L3 + self.avg_t_L4
        )

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
        # 二级/三级：为仅3D与仅2D检测计算外观特征（若未关闭embedding且允许使用外观）
        if not self.embedding_off and not self.app_off and len(detection_3D_only) > 0:
            det_3D_only_bboxs = [det_3d_o.additional_info[2:6] for det_3d_o in detection_3D_only]
            dets_3D_only_embs = self.embedder.compute_embedding(img, det_3D_only_bboxs)
        if not self.embedding_off and not self.app_off and len(detection_2D_only) > 0:
            det_2D_only_bboxs = [det.to_x1y1x2y2() for det in detection_2D_only]
            dets_2D_only_embs = self.embedder.compute_embedding(img, det_2D_only_bboxs)
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

        try:
            if len(detection_3D_fusion_conf) == len(detection_3D_fusion):
                for i, det in enumerate(detection_3D_fusion):
                    try:
                        det.score = float(detection_3D_fusion_conf[i])
                    except Exception:
                        pass
        except Exception:
            pass

        # 更新帧计数 (在关联之前)
        self.current_frame += 1
        self.last_t_L15 = 0.0
        self.last_t_L25 = 0.0

        # 1st Level of Association
        t0 = time.time()
        iou_shreshold=0.01
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_detections_to_trackers_fusion(
            detection_3D_fusion, self.tracks_3d, self.aw_off, self.grid_off, self.mot_off, iou_shreshold,
            det_embs=dets_3D_fusion_embs, det_app=False, appearance_weight=self.appearance_weight_level1,
            heading_options=self._get_heading_options(stage='l1'))
        
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(detection_3D_fusion[detection_idx], current_frame=self.current_frame)
            if use_app_L1 and dets_3D_fusion_embs.shape[0] > detection_idx:
                # 软更新策略：低分数采用极小步长
                alpha_use = None
                try:
                    sc = float(getattr(detection_3D_fusion[detection_idx], 'score', 1.0))
                except Exception:
                    sc = 1.0
                if sc < 0.4:
                    alpha_use = 0.99
                else:
                    if dets_fusion_alpha is not None and dets_fusion_alpha.shape[0] > detection_idx:
                        alpha_use = float(dets_fusion_alpha[detection_idx])
                    else:
                        alpha_use = self.alpha_fixed_emb
                self.tracks_3d[track_idx].update_emb(dets_3D_fusion_embs[detection_idx], alpha=alpha_use)
                self._maybe_add_track_memory_feature(
                    self.tracks_3d[track_idx],
                    dets_3D_fusion_embs[detection_idx],
                    det_score=sc,
                )
                try:
                    detection_3D_fusion[detection_idx].feature = self.tracks_3d[track_idx].emb
                except Exception:
                    pass
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
        
        
        self.last_t_L1 = time.time() - t0
        # ========== Level 1.5: 速度自适应回溯关联 ==========
        # 在处理未匹配之前,先尝试速度回溯
        if self.velocity_backtrack_enabled and \
           len(unmatched_dets_fusion_idx) > 0 and \
           len(unmatched_trks_fusion_idx) > 0:
            t1 = time.time()
            
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
                    detection_3D_fusion[original_det_idx],
                    current_frame=self.current_frame
                )
                if not self.app_off:
                    # 软更新策略：低分数采用极小步长
                    try:
                        sc = float(getattr(detection_3D_fusion[original_det_idx], 'score', 1.0))
                    except Exception:
                        sc = 1.0
                    alpha_use = 0.99 if sc < 0.4 else self.alpha_fixed_emb
                    self.tracks_3d[original_trk_idx].update_emb(dets_3D_fusion_embs[original_det_idx], alpha=alpha_use)
                    self._maybe_add_track_memory_feature(
                        self.tracks_3d[original_trk_idx],
                        dets_3D_fusion_embs[original_det_idx],
                        det_score=sc,
                    )
                    try:
                        detection_3D_fusion[original_det_idx].feature = self.tracks_3d[original_trk_idx].emb
                    except Exception:
                        pass
                self.tracks_3d[original_trk_idx].state = 2
                self.tracks_3d[original_trk_idx].fusion_time_update = 0
                
            
            # 更新未匹配列表 (只保留真正未匹配的)
            unmatched_dets_fusion_idx = [
                unmatched_dets_fusion_idx[i] for i in velocity_unmatched_dets
            ]
            unmatched_trks_fusion_idx = [
                unmatched_trks_fusion_idx[i] for i in velocity_unmatched_trks
            ]
            
            if len(velocity_matched) > 0:
                self.total_L15_recoveries += len(velocity_matched)  # 累计统计
            self.last_t_L15 = time.time() - t1
        else:
            pass

        # 处理最终未匹配的轨迹
        for track_idx in unmatched_trks_fusion_idx:
            self.tracks_3d[track_idx].fusion_time_update += 1
            self.tracks_3d[track_idx].mark_missed()

        #  2nd Level of Association
        self.unmatch_tracks_3d1 = [t for t in self.tracks_3d if t.time_since_update > 0]
        t2 = time.time()
        iou_shreshold=0.01
        matched_only_idx, unmatched_dets_only_idx, _ = associate_detections_to_trackers_fusion(
            detection_3D_only, self.unmatch_tracks_3d1, self.aw_off, self.grid_off, self.mot_off, iou_shreshold,
            det_embs=dets_3D_only_embs, det_app=self.app_off, appearance_weight=self.appearance_weight_level1,
            heading_options=self._get_heading_options(stage='l2'))
        index_to_delete = []
        for detection_idx, track_idx in matched_only_idx:
            for index, t in enumerate(self.tracks_3d):
                if t.track_id_3d == self.unmatch_tracks_3d1[track_idx].track_id_3d:
                    t.update_3d(detection_3D_only[detection_idx], current_frame=self.current_frame)
                    if not self.app_off:
                        # 软更新策略：低分数采用极小步长（若无score则按固定alpha）
                        try:
                            sc = float(getattr(detection_3D_only[detection_idx], 'score', 1.0))
                        except Exception:
                            sc = 1.0
                        alpha_use = 0.99 if sc < 0.4 else self.alpha_fixed_emb
                        t.update_emb(dets_3D_only_embs[detection_idx], alpha=alpha_use)
                        self._maybe_add_track_memory_feature(
                            t,
                            dets_3D_only_embs[detection_idx],
                            det_score=sc,
                        )
                    index_to_delete.append(track_idx)
                    break
        self.unmatch_tracks_3d1 = [self.unmatch_tracks_3d1[i] for i in range(len(self.unmatch_tracks_3d1)) if i not in index_to_delete]
        for detection_idx in unmatched_dets_only_idx:
            self._initiate_track_3d(detection_3D_only[detection_idx], dets_3D_only_embs[detection_idx])
        self.last_t_L2 = time.time() - t2

        # ========== 多帧关联 (Level 2.5) - 移至L2之后 ==========
        # 在L2（仅3D）之后，对仍未匹配的轨迹进行多帧历史关联
        if self.enable_backtrack_global and \
           self.multi_frame_config.enable_multi_frame_backtrack and \
           len(self.unmatch_tracks_3d1) > 0:
            t25 = time.time()

            unmatched_trks_mf = list(self.unmatch_tracks_3d1)

            # ========== 使用深度学习增强的回溯（如果启用） ==========
            if hasattr(self, 'use_dl_backtrack') and self.use_dl_backtrack:
                from backtrack_depth_learning.dl_backtrack_integration import dl_enhanced_backtrack_association
                multi_frame_matches = dl_enhanced_backtrack_association(
                    unmatched_trks_mf,
                    self.detection_history,
                    self.current_frame,
                    self.multi_frame_config,
                    predictor=getattr(self, 'dl_predictor', None)
                )
            else:
                # 使用原始方法
                multi_frame_matches = multi_frame_backtrack_association(
                    unmatched_trks_mf,
                    self.detection_history,
                    self.current_frame,
                    self.multi_frame_config
                )

            if len(multi_frame_matches) > 0:
                updated_tracks = process_multi_frame_matches(
                    multi_frame_matches,
                    virtual_update_config=self.multi_frame_config,
                    current_frame=self.current_frame,
                    verbose=self.multi_frame_config.verbose
                )
                l25_initial_id_map = {
                    id(trk): int(getattr(trk, 'track_id_3d', -1))
                    for trk in updated_tracks
                }

                # 从未匹配列表中移除已匹配的轨迹（按ID）
                recovered_ids = set(t.track_id_3d for t in updated_tracks)
                self.unmatch_tracks_3d1 = [t for t in self.unmatch_tracks_3d1 if t.track_id_3d not in recovered_ids]

                # 可选：更新外观（基于当前帧融合3D检测近邻）
                if not self.app_off and len(updated_tracks) > 0:
                    for trk in updated_tracks:
                        for i, det in enumerate(detection_3D_fusion):
                            if np.allclose(det.bbox[:3], trk.pose[:3], atol=0.1):
                                # 软更新策略：低分数采用极小步长
                                try:
                                    sc = float(getattr(det, 'score', 1.0))
                                except Exception:
                                    sc = 1.0
                                alpha_use = 0.99 if sc < 0.4 else self.alpha_fixed_emb
                                trk.update_emb(dets_3D_fusion_embs[i], alpha=alpha_use)
                                break

                self.total_L25_recoveries += len(multi_frame_matches)  # 累计统计
            else:
                pass
            self.last_t_L25 = time.time() - t25
        # 在L2.5之后再对未匹配的融合3D检测新建轨迹
        if len(unmatched_dets_fusion_idx) > 0:
            for detection_idx in unmatched_dets_fusion_idx:
                self._initiate_track_3d(detection_3D_fusion[detection_idx], dets_3D_fusion_embs[detection_idx])

        # 冲突轨迹清理（基于中心距离）
        def _cleanup_track_conflicts(pos_thresh=1.0):
            active = [t for t in self.tracks_3d if not t.is_deleted()]
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    ti = active[i]
                    tj = active[j]
                    try:
                        d = float(np.linalg.norm(ti.pose[:3] - tj.pose[:3]))
                    except Exception:
                        continue
                    if d <= pos_thresh:
                        ai = ti.is_confirmed()
                        aj = tj.is_confirmed()
                        if ai and not aj:
                            loser = tj
                        elif aj and not ai:
                            loser = ti
                        else:
                            if ti.hits != tj.hits:
                                loser = ti if ti.hits < tj.hits else tj
                            elif ti.age != tj.age:
                                loser = ti if ti.age < tj.age else tj
                            else:
                                loser = ti if ti.track_id_3d > tj.track_id_3d else tj
                        loser.state = TrackState.Deleted

        _cleanup_track_conflicts(pos_thresh=1.0)

        self.unmatch_tracks_3d2 = [t for t in self.tracks_3d if t.time_since_update == 0 and t.hits == 1 ]
        self.unmatch_tracks_3d = self.unmatch_tracks_3d1 + self.unmatch_tracks_3d2

        # 3rd Level of Association
        t3 = time.time()
        iou_shreshold=0.4
        matched, unmatch_trks, unmatch_dets = associate_detections_to_tracks(self.tracks_2d, detection_2D_only, iou_shreshold, self.aw_off,self.grid_off,self.mot_off, det_embs=dets_2D_only_embs, det_app = self.app_off)
        for track_idx, detection_idx in matched:
            self.tracks_2d[track_idx].update_2d(self.kf_2d, detection_2D_only[detection_idx])
            if not self.app_off:
                self.tracks_2d[track_idx].update_emb(dets_2D_only_embs[detection_idx])
        for track_idx in unmatch_trks:
            self.tracks_2d[track_idx].mark_missed()
        for detection_idx in unmatch_dets:
            self._initiate_track_2d(detection_2D_only[detection_idx], dets_2D_only_embs[detection_idx])
        self.tracks_2d = [t for t in self.tracks_2d if not t.is_deleted()]
        self.last_t_L3 = time.time() - t3

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
        self.last_t_L4 = time.time() - t4
        self._accumulate_timing()
        
        # ========== DEBUG: 检查重复ID ==========
        if 'updated_tracks' in locals() and len(updated_tracks) > 0:
            for trk in updated_tracks:
                self._append_l25_final_hit_log(
                    trk,
                    l25_initial_id_map.get(id(trk), getattr(trk, 'track_id_3d', -1))
                )
        track_ids = [t.track_id_3d for t in self.tracks_3d if t.is_confirmed()]
        if len(track_ids) != len(set(track_ids)):
            from collections import Counter
            duplicates = [id for id, count in Counter(track_ids).items() if count > 1]
            print(f"❌ 警告：发现重复的轨迹ID: {duplicates}")
            print(f"   所有ID: {track_ids}")
            print(f"   帧号: {self.current_frame}")
        # ========================================
        
        # 更新检测历史 (帧计数已在函数开始时更新)
        # 将当前帧融合3D检测的外观特征注入历史，便于回溯阶段使用外观相似度
        try:
            if (
                isinstance(detection_3D_fusion, (list, tuple))
                and len(detection_3D_fusion) == dets_3D_fusion_embs.shape[0]
            ):
                for i, det in enumerate(detection_3D_fusion):
                    try:
                        if getattr(det, 'feature', None) is None:
                            det.feature = dets_3D_fusion_embs[i]
                    except Exception:
                        pass
        except Exception:
            pass
        self._update_detection_history(detection_3D_fusion)
        self._debug_print_active_betas()

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
                base_score = (
                    w_vel_t * velocity_matrix[d, t] +
                    w_pos_t * position_matrix[d, t]
                )
                if self.use_rotated_geom_in_l15:
                    rotated_geom_sim = compute_rotated_ground_similarity(
                        detections[d].bbox,
                        tracks[t].pose,
                    )
                    geom_w = max(0.0, min(1.0, float(self.rotated_geom_weight_l15)))
                    base_score = (1.0 - geom_w) * base_score + geom_w * rotated_geom_sim
                if self.use_state_heading_in_l15:
                    heading_score = self._compute_state_heading_consistency(
                        tracks[t], detections[d]
                    )
                    heading_mix = max(0.0, min(1.0, 0.5 * (
                        float(getattr(tracks[t], 'heading_conf_det', 0.5)) +
                        float(getattr(tracks[t], 'heading_conf_vel', 0.5))
                    )))
                    combined_matrix[d, t] = (1.0 - heading_mix) * base_score + heading_mix * heading_score
                else:
                    combined_matrix[d, t] = base_score
        
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
            weight: 自适应角度权重（用于角度特征）
        """
        if not self.adaptive_weight:
            return 1.0  # 不启用自适应时，返回满权重
        
        # 获取轨迹速度
        velocity = get_velocity(track)
        speed = np.linalg.norm(velocity)
        
        # 归一化速度到[0, 1]
        v_norm = speed / self.velocity_weight_vmax
        
        gamma_min = 0.20

        # 速度自适应权重抑制策略 γ(v)
        # γ(v) = max(γ_min, ((v - v_low) / (v_high - v_low))^p)
        if v_norm < self.adaptive_threshold_low:
            # 低速：抑制角度权重到最小值
            return gamma_min
        elif v_norm > self.adaptive_threshold_high:
            # 高速：完全信任角度信息
            return 1.0
        else:
            # 中速：指数形式的权重过渡
            ratio = (v_norm - self.adaptive_threshold_low) / \
                    (self.adaptive_threshold_high - self.adaptive_threshold_low)
            p = 2.0  # 指数参数（凸曲线）
            gamma = gamma_min + (1.0 - gamma_min) * (ratio ** p)  # 从gamma_min到1.0的指数增长
            return gamma

    def _update_detection_history(self, detections):
        """
        更新检测历史 (用于速度估计)
        """
        self.detection_history[self.current_frame] = detections
        
        # 只保留最近5帧
        if len(self.detection_history) > 5:
            oldest_frame = min(self.detection_history.keys())
            del self.detection_history[oldest_frame]

    def reset_timing_stats(self):
        self.sum_t_L1 = 0.0
        self.cnt_t_L1 = 0
        self.sum_t_L15 = 0.0
        self.cnt_t_L15 = 0
        self.sum_t_L2 = 0.0
        self.cnt_t_L2 = 0
        self.sum_t_L25 = 0.0
        self.cnt_t_L25 = 0
        self.sum_t_L3 = 0.0
        self.cnt_t_L3 = 0
        self.sum_t_L4 = 0.0
        self.cnt_t_L4 = 0

    def _accumulate_timing(self):
        e = self._timing_eps
        if self.last_t_L1 > e:
            self.sum_t_L1 += self.last_t_L1
            self.cnt_t_L1 += 1
        if self.last_t_L15 > e:
            self.sum_t_L15 += self.last_t_L15
            self.cnt_t_L15 += 1
        if self.last_t_L2 > e:
            self.sum_t_L2 += self.last_t_L2
            self.cnt_t_L2 += 1
        if self.last_t_L25 > e:
            self.sum_t_L25 += self.last_t_L25
            self.cnt_t_L25 += 1
        if self.last_t_L3 > e:
            self.sum_t_L3 += self.last_t_L3
            self.cnt_t_L3 += 1
        if self.last_t_L4 > e:
            self.sum_t_L4 += self.last_t_L4
            self.cnt_t_L4 += 1

    def _debug_print_active_betas(self):
        return

    def _initiate_track_3d(self, detection,emb=None):
        self.kf_3d = KalmanBoxTracker(detection.bbox)
        self.additional_info = detection.additional_info
        pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.tracks_3d.append(Track_3D(pose, self.kf_3d, self.track_id_3d, self.n_init, self.max_age, self.additional_info, emb, init_frame=self.current_frame))
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
