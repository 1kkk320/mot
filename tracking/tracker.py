import numpy as np
from tracking import kalman_filter_2d
from tracking.cost_function import iou_batch
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
        self.alpha_fixed_emb = 0.95
        self.grid_off = grid_off
        self.app_off = app_off
        self.mot_off = False
        self.embedder = EmbeddingComputer(grid_off=self.grid_off)

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

        # 1st Level of Association
        matched_fusion_idx, unmatched_dets_fusion_idx, unmatched_trks_fusion_idx = associate_detections_to_trackers_fusion(
            detection_3D_fusion, self.tracks_3d,self.aw_off,self.grid_off,self.mot_off, iou_threshold,det_embs=dets_3D_fusion_embs, det_app = self.app_off)
        for detection_idx, track_idx in matched_fusion_idx:
            self.tracks_3d[track_idx].update_3d(detection_3D_fusion[detection_idx])
            if not self.app_off:
                self.tracks_3d[track_idx].update_emb(dets_3D_fusion_embs[detection_idx])
            self.tracks_3d[track_idx].state = 2
            self.tracks_3d[track_idx].fusion_time_update = 0
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
                        # print(self.tracks_3d[i].track_id_3d,self.tracks_2d[track_idx_2d].track_id_2d)
                        self.tracks_3d[i].track_id_3d = self.tracks_2d[track_idx_2d].track_id_2d
                        # print("recite:",self.tracks_3d[i].track_id_3d)
                    self.tracks_3d[i].time_since_update = 0
                    if self.tracks_2d[track_idx_2d].hits >= 2:
                        self.tracks_3d[i].hits = self.tracks_2d[track_idx_2d].hits + 1
                    else:
                        self.tracks_3d[i].hits += 1
                    self.tracks_3d[i].state_update()
            index_to_delete2.append(track_idx_2d)
        self.tracks_2d = [self.tracks_2d[i] for i in range(len(self.tracks_2d)) if i not in index_to_delete2]
        self.tracks_3d = [t for t in self.tracks_3d if not t.is_deleted()]

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