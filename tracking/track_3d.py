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
        self.con1 = 0.97
        self.emb = emb

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