import numpy as np

'''
  3D track management
  Reactivate: When a confirmed trajectory is occluded and in turn cannot be associated with any detections for several frames, it 
  is then regarded as a reappeared trajectory.
'''

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Reactivate = 4

class TrackState3Dor2D:
    Tracking_3D = 1
    Tracking_2D = 2


class Track_3D:
    def __init__(self, pose, kf_3d, track_id_3d, n_init, max_age,additional_info,emb = None, feature=None):
        self.pose = pose
        self.kf_3d = kf_3d
        self.track_id_3d = track_id_3d
        self.hits = 1
        self.age = 1
        self.state = TrackState.Tentative
        self.n_init = n_init
        self._max_age = max_age
        self.is3D_or_2D_track = TrackState3Dor2D.Tracking_3D
        self.additional_info = additional_info
        self.time_since_update = 0
        self.fusion_time_update = 0
        self.confidence = 0.97
        self.con1 = 0.96
        self.emb = emb
        
        # ========== 方案3: 多帧回溯 - 速度历史 ==========
        self.velocity_history = []  # [(frame_id, velocity), ...]
        self.max_history_length = 5  # 保留最近5帧
        
        # 初始化第一帧速度
        initial_velocity = self.kf_3d.kf.x[7:10].flatten()
        self.velocity_history.append((self.age, initial_velocity.copy()))

    def predict_3d(self, trk_3d):
        self.pose = trk_3d.predict()

    def update_3d(self, detection_3d):
        self.kf_3d.update(detection_3d.bbox)
        self.additional_info = detection_3d.additional_info
        self.pose = np.concatenate(self.kf_3d.kf.x[:7], axis=0)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.confidence = 1
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative
        if  self.fusion_time_update >= 3:
            self.state = TrackState.Reactivate
        
        # ========== 方案3: 记录速度历史 ==========
        current_velocity = self.kf_3d.kf.x[7:10].flatten()
        self.velocity_history.append((self.age, current_velocity.copy()))
        
        # 保持历史长度
        if len(self.velocity_history) > self.max_history_length:
            self.velocity_history.pop(0)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        '''
        对更新后的嵌入向量进行归一化，以确保它具有单位范数（向量的长度为1）。
        这样做可以使嵌入向量更容易用于相似性比较，因为它们在尺度上是一致的。
        '''
        self.emb /= np.linalg.norm(self.emb)

    def state_update(self):
        if self.hits >= self.n_init:
            self.state = TrackState.Confirmed
        else:
            self.state = TrackState.Tentative

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.Confirmed and self.hits >= self.n_init:
            self.state = TrackState.Reactivate
        elif self.time_since_update >= 1 and self.state != TrackState.Reactivate:
            self.state = TrackState.Deleted
        elif self.state == TrackState.Reactivate and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        if self.state == TrackState.Reactivate:
            if self.time_since_update > 1:
                # print(self.time_since_update)
                self.confidence *= self.con1**self.time_since_update

    def fusion_state(self):
        if  self.fusion_time_update >= 2:
            self.state = TrackState.Deleted

    def is_deleted(self):
        return self.state == TrackState.Deleted

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_track_id_3d(self):
        track_id_3d= self.track_id_3d
        return track_id_3d

    def get_emb(self):
        return self.emb
    
    # ========== 方案3: 多帧回溯 - 新增方法 ==========
    
    def get_velocity_trend(self):
        """
        获取速度趋势 (加速度)
        使用最近两帧计算，降低噪声影响
        
        Returns:
            trend: 速度变化趋势 [dvx, dvy, dvz] (m/s per frame)
        """
        if len(self.velocity_history) < 2:
            return np.zeros(3)
        
        # 使用最近两帧计算趋势
        recent_vel = self.velocity_history[-1][1]
        prev_vel = self.velocity_history[-2][1]
        
        # 速度变化率
        trend = recent_vel - prev_vel
        
        return trend
    
    def get_average_velocity(self, window=3):
        """
        获取平均速度 (平滑，降低噪声)
        
        修复版本: 考虑缺失帧的帧差
        当velocity_history中有缺失帧时，使用加权平均而不是简单平均
        
        Args:
            window: 平均窗口大小 (默认3帧)
        
        Returns:
            avg_velocity: 平均速度 [vx, vy, vz]
        """
        if len(self.velocity_history) == 0:
            return np.zeros(3)
        
        if len(self.velocity_history) < window:
            # 历史不足，返回最后一帧速度
            return self.velocity_history[-1][1]
        
        # 获取最近window帧
        recent_vels = self.velocity_history[-window:]
        
        # 计算实际帧差 (考虑缺失帧)
        frame_indices = [v[0] for v in recent_vels]
        frame_diff = frame_indices[-1] - frame_indices[0]
        
        if frame_diff == 0:
            return recent_vels[-1][1]
        
        # 提取速度向量
        velocities = np.array([v[1] for v in recent_vels])
        
        # 加权平均: 越近的帧权重越高
        weights = np.linspace(1, window, window) / (window * (window + 1) / 2)
        weights = weights.reshape(-1, 1)  # 转换为列向量以支持广播
        
        avg_velocity = np.average(velocities, axis=0, weights=weights.flatten())
        
        return avg_velocity
    
    def get_smooth_velocity_trend(self, window=3):
        """
        获取平滑的速度趋势 (降低噪声)
        
        修复版本: 使用实际帧差计算加速度
        当velocity_history中有缺失帧时，使用实际帧差而不是假设固定帧间隔
        
        Args:
            window: 平滑窗口大小
        
        Returns:
            smooth_trend: 平滑后的速度趋势 (加速度)
        """
        if len(self.velocity_history) < 2:
            return np.zeros(3)
        
        if len(self.velocity_history) < window + 1:
            # 历史不足，使用最近两帧计算
            v_new = self.velocity_history[-1][1]
            v_old = self.velocity_history[0][1]
            frame_new = self.velocity_history[-1][0]
            frame_old = self.velocity_history[0][0]
        else:
            # 使用window大小的窗口
            v_new = self.velocity_history[-1][1]
            v_old = self.velocity_history[-window][1]
            frame_new = self.velocity_history[-1][0]
            frame_old = self.velocity_history[-window][0]
        
        # 计算实际帧差 (考虑缺失帧)
        frame_diff = frame_new - frame_old
        
        if frame_diff == 0:
            return np.zeros(3)
        
        # 加速度 = 速度变化 / 帧差
        # 这样可以正确处理缺失帧的情况
        smooth_trend = (v_new - v_old) / frame_diff
        
        return smooth_trend