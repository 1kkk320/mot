from __future__ import print_function
import os, numpy as np, time, cv2, torch, math
from os import listdir
from os.path import join
from file_operation.file import load_list_from_folder, mkdir_if_inexistence, fileparts
from detection.detection import Detection_2D, Detection_3D_only, Detection_3D_Fusion
from tracking.tracker import Tracker
from tracking.motion_reliability import load_kitti_tracking_gt_by_frame, assign_detection_gt_ids
from datasets.datafusion import datafusion2Dand3D
from datasets.coordinate_transformation import convert_3dbox_to_8corner, convert_x1y1x2y2_to_tlwh
from visualization.visualization_3d import show_image_with_boxes
from visualization.visualization_2d import plot_one_box
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_l15_selective_risk_preset():
    preset_name = os.getenv('L15_RISK_PRESET', 'weaker').strip().lower()
    presets = {
        'weaker': {
            'name': 'weaker',
            'gate_focus_gain': 7.0,
            'margin_center': 0.30,
            'margin_gain': 10.0,
            'min_effect': 0.01,
        },
        'current': {
            'name': 'current',
            'gate_focus_gain': 8.0,
            'margin_center': 0.35,
            'margin_gain': 10.0,
            'min_effect': 0.01,
        },
        'stronger': {
            'name': 'stronger',
            'gate_focus_gain': 9.0,
            'margin_center': 0.40,
            'margin_gain': 10.0,
            'min_effect': 0.01,
        },
    }
    preset = presets.get(preset_name, presets['current']).copy()
    if preset_name not in presets:
        print(f"[L1.5 RiskPreset] Unknown preset '{preset_name}', fallback to 'current'")
    return preset


def get_l15_motion_position_weight_preset():
    # Keep 0.50 / 0.50 as the verified working baseline unless a sweep overrides it.
    preset_name = os.getenv('L15_WEIGHT_PRESET', 'balanced').strip().lower()
    presets = {
        'pos_heavier': {
            'name': 'pos_heavier',
            'motion_weight': 0.45,
            'position_weight': 0.55,
        },
        'balanced': {
            'name': 'balanced',
            'motion_weight': 0.50,
            'position_weight': 0.50,
        },
        'motion_heavier': {
            'name': 'motion_heavier',
            'motion_weight': 0.55,
            'position_weight': 0.45,
        },
        'motion_heaviest': {
            'name': 'motion_heaviest',
            'motion_weight': 0.60,
            'position_weight': 0.40,
        },
    }
    preset = presets.get(preset_name, presets['balanced']).copy()
    if preset_name not in presets:
        print(f"[L1.5 WeightPreset] Unknown preset '{preset_name}', fallback to 'balanced'")
    return preset


def get_l12_defer_preset():
    preset_name = os.getenv('L12_DEFER_PRESET', 'current').strip().lower()
    presets = {
        'current': {
            'name': 'current',
            'defer_threshold': 0.16,
            'identity_floor': 0.60,
        },
        'mild': {
            'name': 'mild',
            'defer_threshold': 0.14,
            'identity_floor': 0.62,
        },
        'moderate': {
            'name': 'moderate',
            'defer_threshold': 0.12,
            'identity_floor': 0.64,
        },
    }
    preset = presets.get(preset_name, presets['current']).copy()
    if preset_name not in presets:
        print(f"[L1 DeferPreset] Unknown preset '{preset_name}', fallback to 'current'")
    return preset


def get_l25_geometry_mode():
    mode = os.getenv('L25_GEOMETRY_MODE', 'box_iou').strip().lower()
    aliases = {
        'box': 'box_iou',
        'box_iou': 'box_iou',
        'iou': 'box_iou',
        'rotated': 'rotated_geom',
        'rotated_geom': 'rotated_geom',
        'rot_geom': 'rotated_geom',
    }
    resolved = aliases.get(mode, 'box_iou')
    if mode not in aliases:
        print(f"[L2.5 Geometry] Unknown mode '{mode}', fallback to 'box_iou'")
    return resolved


def get_env_flag(name, default='0'):
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def get_env_int(name, default=None):
    raw = os.getenv(name, '').strip()
    if raw == '':
        return default
    try:
        return int(raw)
    except Exception:
        return default


def get_env_list(name):
    raw = os.getenv(name, '').strip()
    if raw == '':
        return None
    items = [item.strip() for item in raw.split(',') if item.strip()]
    return set(items) if items else None


def _normalize_triplet_weights(bev_weight, center_weight, size_weight):
    bev_weight = max(float(bev_weight), 0.0)
    center_weight = max(float(center_weight), 0.0)
    size_weight = max(float(size_weight), 0.0)
    total = bev_weight + center_weight + size_weight
    if total <= 1e-6:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return bev_weight / total, center_weight / total, size_weight / total


def get_tracker_output_name(data_root):
    parts = set(os.path.normpath(data_root).split(os.sep))
    if 'test' in parts or 'testing' in parts:
        return 'virconv_OCM_test'
    return 'virconv_OCM'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    不同id给予不同的颜色
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


class DeepFusion(object):
    def __init__(self, max_age, min_hits,iou_shreshold=0.4):
        '''
        :param max_age:  The maximum frames in which an object disappears.消除目标轨迹的最大帧数
        :param min_hits: The minimum frames in which an object becomes a trajectory in succession.成为轨迹的最小帧数
        '''
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracker = Tracker(max_age, min_hits, grid_off=True, app_off=True)
        self.reorder = [3, 4, 5, 6, 2, 1, 0]
        self.reorder_back = [6, 5, 4, 0, 1, 2, 3]
        self.frame_count = 0
        self.iou_shreshold = iou_shreshold
        self.current_gt_frame_entries = []

    def update(self,detection_3D_fusion,detection_2D_only,detection_3D_only,detection_3Dto2D_only,
               additional_info, calib_file,img,detection_2D_only_conf, detection_3D_fusion_conf):

        dets_3d_fusion = np.array(detection_3D_fusion['dets_3d_fusion'])
        dets_3d_fusion_info = np.array(detection_3D_fusion['dets_3d_fusion_info'])
        dets_3d_only = np.array(detection_3D_only['dets_3d_only'])
        dets_3d_only_info = np.array(detection_3D_only['dets_3d_only_info'])

        if len(dets_3d_fusion) == 0:
            dets_3d_fusion = dets_3d_fusion
        else:
            dets_3d_fusion = dets_3d_fusion[:,self.reorder]  # convert [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h]
        if len(dets_3d_only) == 0:
            dets_3d_only = dets_3d_only
        else:
            dets_3d_only = dets_3d_only[:, self.reorder] # convert [h,w,l,x,y,z,rot_y] to [x,y,z,rot_y，l,w,h]

        detection_3D_fusion = [Detection_3D_Fusion(det_fusion, dets_3d_fusion_info[i]) for i, det_fusion in enumerate(dets_3d_fusion)]
        detection_3D_only = [Detection_3D_only(det_only, dets_3d_only_info[i]) for i, det_only in enumerate(dets_3d_only)]
        detection_2D_only = [Detection_2D(det_fusion) for i, det_fusion in enumerate(detection_2D_only)]
        if self.current_gt_frame_entries:
            assign_detection_gt_ids(detection_3D_fusion, self.current_gt_frame_entries, min_iou=0.5)
            assign_detection_gt_ids(detection_3D_only, self.current_gt_frame_entries, min_iou=0.5)

        self.tracker.predict_2d()
        self.tracker.predict_3d()
        self.tracker.update(detection_3D_fusion, detection_3D_only, detection_3Dto2D_only, detection_2D_only, calib_file, img,detection_2D_only_conf, detection_3D_fusion_conf, self.iou_shreshold)

        self.frame_count += 1
        outputs = []
        output_meta = []
        for track in self.tracker.tracks_3d:
            if track.is_confirmed():
                bbox = np.array(track.pose[self.reorder_back])
                outputs.append(np.concatenate(([track.track_id_3d], bbox, track.additional_info)).reshape(1, -1))
                output_meta.append({
                    'track_id': int(track.track_id_3d),
                    'assoc_level': str(getattr(track, 'last_assoc_level', 'UNKNOWN')),
                    'assoc_frame': int(getattr(track, 'last_assoc_frame', -1)),
                    'time_since_update': int(getattr(track, 'time_since_update', -1)),
                    'hits': int(getattr(track, 'hits', -1)),
                })
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
            # print(outputs)
        """提取出跟踪的2d轨迹"""
        outputs_2d = []
        for track in self.tracker.tracks_2d:
            if track.is_confirmed():
                bbox_2d = np.array(np.array(track.x1y1x2y2()))
                # print(bbox_2d)
                outputs_2d.append(np.concatenate(([track.track_id_2d],bbox_2d)).reshape(1,-1))
        if len(outputs_2d) > 0:
            outputs_2d = np.stack(outputs_2d,axis=0)
            # print("2d轨迹",outputs_2d,type(outputs_2d))


        return outputs, outputs_2d, output_meta

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):  # Convert the coordinate format of the bbox box from center x, y, w, h to upper left x, upper left y, w, h
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), 0)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), 0)
        return x1, y1, x2, y2

    def _tlwh_to_x1y1x2y2(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return x1, y1, x2, y2


def main():
    l15_risk_preset = get_l15_selective_risk_preset()
    l15_weight_preset = get_l15_motion_position_weight_preset()
    l12_defer_preset = get_l12_defer_preset()
    l25_geometry_mode = get_l25_geometry_mode()
    # Define the file name
    data_root = 'datasets/kitti/train'
    detections_name_3D = '3D_virconv'  
    detections_name_2D = '2D_rrc_Car'  

    # Define the file path
    calib_root = os.path.join(data_root, 'calib')     #矫正数据
    dataset_dir = os.path.join(data_root,'image_02')
    detections_root_3D = os.path.join(data_root, detections_name_3D)
    detections_root_2D = os.path.join(data_root, detections_name_2D)
    parts = set(os.path.normpath(data_root).split(os.sep))
    tracker_output_name = get_tracker_output_name(data_root)
    save_root = os.path.join(r'E:\mot\results', tracker_output_name)
    txt_path_0 = os.path.join(save_root, 'data'); mkdir_if_inexistence(txt_path_0)
    image_path_0 = os.path.join(save_root, 'image'); mkdir_if_inexistence(image_path_0)
    assoc_diag_path_0 = os.path.join(save_root, 'assoc_levels'); mkdir_if_inexistence(assoc_diag_path_0)
    # Open file to save in list.打开保存在列表里面的文件
    det_id2str = {1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}
    calib_files = os.listdir(calib_root) #返回指定的文件夹包含的文件或文件夹的名字的列表。
    detections_files_3D = os.listdir(detections_root_3D)
    detections_files_2D = os.listdir(detections_root_2D)
    all_image_files = os.listdir(dataset_dir)
    image_files = list(all_image_files)
    detection_file_list_3D, num_seq_3D = load_list_from_folder(detections_files_3D, detections_root_3D)
    detection_file_list_2D, num_seq_2D = load_list_from_folder(detections_files_2D, detections_root_2D)
    image_file_list, _ = load_list_from_folder(image_files, dataset_dir)

    # Pre-create empty result files for evaluator/consistency
    if 'train' in parts or 'training' in parts:
        all_train_seqs = [f"{i:04d}" for i in range(21)]
        for name in all_train_seqs:
            open(os.path.join(txt_path_0, name + '.txt'), 'w').close()
    else:
        for name in all_image_files:
            open(os.path.join(txt_path_0, name + '.txt'), 'w').close()

    total_time, total_frames, i = 0.0, 0, 0  # Tracker runtime, total frames and Serial number of the dataset、跟踪器运行时，总时间、总帧数和数据集的序列号
    tracker = DeepFusion(max_age=25, min_hits=3, iou_shreshold=0.22)
    
    # 保持外观权重
    tracker.tracker.appearance_weight_level1 = float(os.getenv('L12_APPEARANCE_WEIGHT', '0.10'))

    # ========== 关闭航向角创新点 ==========
    enable_l12_geom = get_env_flag('ENABLE_L12_GEOM', '1')
    enable_motion_rel = get_env_flag('ENABLE_MOTION_REL', '1')
    enable_l4_takeover = get_env_flag('ENABLE_L4_TAKEOVER', '1')
    # Paper-style ablation semantics:
    # - L1/L2 is always the base trunk.
    # - "MotionRel" denotes the whole L1.5/L2.5 recovery branch rather than
    #   only a small reweighting sub-switch inside an always-on branch.
    # - L4 base association remains active; ENABLE_L4_TAKEOVER only controls
    #   whether identity takeover is allowed after a valid L4 match.
    enable_recovery_branch = enable_motion_rel
    tracker.tracker.use_rotated_geom_in_l1 = enable_l12_geom
    tracker.tracker.use_rotated_geom_in_l2 = enable_l12_geom
    tracker.tracker.use_rotated_geom_in_l15 = enable_recovery_branch
    tracker.tracker.rotated_geom_weight_l15 = float(os.getenv('L15_ROTATED_GEOM_WEIGHT', '0.20'))
    bev_w, center_w, size_w = _normalize_triplet_weights(
        os.getenv('L12_ROT_GEOM_BEV_WEIGHT', '0.333333'),
        os.getenv('L12_ROT_GEOM_CENTER_WEIGHT', '0.333333'),
        os.getenv('L12_ROT_GEOM_SIZE_WEIGHT', '0.333333'),
    )
    tracker.tracker.l12_rotated_geom_bev_weight = bev_w
    tracker.tracker.l12_rotated_geom_center_weight = center_w
    tracker.tracker.l12_rotated_geom_size_weight = size_w
    tracker.tracker.l12_rotated_geom_center_tau = float(os.getenv('L12_ROT_GEOM_CENTER_TAU', '1.0'))
    tracker.tracker.l12_rotated_geom_size_tau = float(os.getenv('L12_ROT_GEOM_SIZE_TAU', '1.0'))
    tracker.tracker.enable_l12_risk_model = True
    tracker.tracker.enable_l12_risk_tempering = False
    tracker.tracker.enable_l1_deferred_commitment = True
    tracker.tracker.enable_l2_deferred_commitment = False
    tracker.tracker.enable_l12_defer_diag_log = True
    tracker.tracker.l12_defer_diag_log_path = os.path.join('logs', 'l12_defer_diag.log')
    tracker.tracker.l12_risk_margin_center = 0.08
    tracker.tracker.l12_risk_app_rescue = 0.06
    tracker.tracker.l12_defer_threshold = l12_defer_preset['defer_threshold']
    tracker.tracker.l12_defer_identity_floor = l12_defer_preset['identity_floor']
    print('[Association Config] RotGeom(L1={}, L2={}) weights=({:.3f},{:.3f},{:.3f}) taus=({:.3f},{:.3f})'.format(
        tracker.tracker.use_rotated_geom_in_l1,
        tracker.tracker.use_rotated_geom_in_l2,
        tracker.tracker.l12_rotated_geom_bev_weight,
        tracker.tracker.l12_rotated_geom_center_weight,
        tracker.tracker.l12_rotated_geom_size_weight,
        tracker.tracker.l12_rotated_geom_center_tau,
        tracker.tracker.l12_rotated_geom_size_tau,
    ))
    print(
        '[Association Weight] appearance_l12={} rotated_geom_l15={}'.format(
            tracker.tracker.appearance_weight_level1,
            tracker.tracker.rotated_geom_weight_l15,
        )
    )
    print(
        '[L1/L2 RiskModel] enabled={} tempering={} defer_l1={} defer_l2={} margin_center={}'.format(
            tracker.tracker.enable_l12_risk_model,
            tracker.tracker.enable_l12_risk_tempering,
            tracker.tracker.enable_l1_deferred_commitment,
            tracker.tracker.enable_l2_deferred_commitment,
            tracker.tracker.l12_risk_margin_center,
        )
    )
    print(
        '[L1 DeferDiag] enabled={} path={}'.format(
            tracker.tracker.enable_l12_defer_diag_log,
            tracker.tracker.l12_defer_diag_log_path,
        )
    )
    print(
        '[L1 DeferPreset] name={} defer_threshold={} identity_floor={}'.format(
            l12_defer_preset['name'],
            tracker.tracker.l12_defer_threshold,
            tracker.tracker.l12_defer_identity_floor,
        )
    )
    print('[Output] tracker_name={} save_root={}'.format(tracker_output_name, save_root))

    # 优化速度自适应参数
    tracker.tracker.adaptive_threshold_low = 0.40  # 0.40
    tracker.tracker.adaptive_threshold_high = 0.70  # 0.70

    # ========== 仅打开 L1.5 速度回溯 ==========
    tracker.tracker.velocity_backtrack_enabled = enable_recovery_branch
    tracker.tracker.velocity_threshold = 0.6  # 速度相似度阈值
    tracker.tracker.adaptive_weight = True  # 启用自适应速度权重
    tracker.tracker.l15_use_fixed_motion_position_weights = True
    tracker.tracker.l15_fixed_motion_weight = l15_weight_preset['motion_weight']
    tracker.tracker.l15_fixed_position_weight = l15_weight_preset['position_weight']
    tracker.tracker.velocity_weight_vmax = 12.0  # 速度归一化最大值
    tracker.tracker.use_velocity_trend = True  # 启用速度趋势
    tracker.tracker.use_smooth_velocity = True  # 启用速度平滑
    tracker.tracker.velocity_smooth_window = 3  # 速度平滑窗口
    tracker.tracker.trend_weight = 0.3  # 趋势权重
    tracker.tracker.use_confidence_aware_motion_l15 = enable_recovery_branch
    tracker.tracker.l15_motion_reliability_mode = os.getenv('L15_MOTION_RELIABILITY_MODE', 'manual').strip().lower()
    tracker.tracker.l15_motion_reliability_model_path = os.getenv('L15_MOTION_RELIABILITY_MODEL', '').strip()
    tracker.tracker.enable_l15_motion_sample_export = os.getenv('EXPORT_L15_MOTION_SAMPLES', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    tracker.tracker.l15_motion_sample_export_path = os.getenv('L15_MOTION_SAMPLE_EXPORT_PATH', os.path.join('logs', 'l15_motion_reliability_samples.jsonl')).strip()
    tracker.tracker.l15_motion_reliability_bias = -0.2
    tracker.tracker.l15_motion_reliability_score_gain = 2.5
    tracker.tracker.l15_motion_reliability_uncertainty_gain = 2.0
    tracker.tracker.l15_motion_reliability_neutral = 0.5
    tracker.tracker.enable_l15_reliability_pair_reweight = enable_recovery_branch and os.getenv('L15_RELIABILITY_PAIR_REWEIGHT', '1').strip().lower() in ('1', 'true', 'yes', 'on')
    tracker.tracker.l15_reliability_pair_reweight_strength = float(os.getenv('L15_RELIABILITY_PAIR_REWEIGHT_STRENGTH', '1.0'))
    tracker.tracker.l15_motion_v4_high_sim_threshold = 0.8
    tracker.tracker.l15_motion_v4_raw_focus_gain = 10.0
    tracker.tracker.l15_motion_v4_gate_focus_gain = l15_risk_preset['gate_focus_gain']
    tracker.tracker.l15_motion_v4_margin_center = l15_risk_preset['margin_center']
    tracker.tracker.l15_motion_v4_margin_gain = l15_risk_preset['margin_gain']
    tracker.tracker.l15_motion_v4_min_effect = l15_risk_preset['min_effect']
    tracker.tracker.enable_l15_motion_diag_log = enable_recovery_branch
    tracker.tracker.l15_motion_diag_log_path = os.path.join(
        'logs', f"l15_motion_diag_{l15_risk_preset['name']}_{l15_weight_preset['name']}.log"
    )
    print(
        "[L1.5 WeightPreset] name={} motion_weight={} position_weight={}".format(
            l15_weight_preset['name'],
            tracker.tracker.l15_fixed_motion_weight,
            tracker.tracker.l15_fixed_position_weight,
        )
    )
    print(
        "[L1.5 RiskPreset] name={} gate_focus_gain={} margin_center={} margin_gain={} min_effect={}".format(
            l15_risk_preset['name'],
            tracker.tracker.l15_motion_v4_gate_focus_gain,
            tracker.tracker.l15_motion_v4_margin_center,
            tracker.tracker.l15_motion_v4_margin_gain,
            tracker.tracker.l15_motion_v4_min_effect,
        )
    )
    
    # ========== 基线：关闭 L2.5 多帧回溯 ==========
    tracker.tracker.multi_frame_config.enable_multi_frame_backtrack = enable_recovery_branch
    tracker.tracker.multi_frame_config.min_backtrack_age = 4  # 最小回溯年龄
    tracker.tracker.multi_frame_config.max_backtrack_age = 15  # 最大回溯年龄
    tracker.tracker.multi_frame_config.lambda_decay = 0.15  # 时间衰减系数
    tracker.tracker.multi_frame_config.cost_threshold = -0.35  # 代价阈值
    tracker.tracker.multi_frame_config.last_k_frames = 5  # 回溯帧数
    tracker.tracker.multi_frame_config.detection_buffer_size = 5  # 检测缓冲大小
    tracker.tracker.multi_frame_config.topk_per_frame = 1  # 每帧top-k
    tracker.tracker.multi_frame_config.appearance_weight = 0.2  # 外观权重
    tracker.tracker.multi_frame_config.appearance_hard_gate = 0.50  # 外观硬门控
    tracker.tracker.multi_frame_config.uncertainty_norm = float(os.getenv('L25_UNCERTAINTY_NORM', '12.0'))
    tracker.tracker.multi_frame_config.velocity_confidence_floor = float(os.getenv('L25_VELOCITY_CONF_FLOOR', '0.25'))
    tracker.tracker.multi_frame_config.enable_confidence_aware_motion_l25 = enable_recovery_branch and os.getenv('L25_ENABLE_MOTION_RELIABILITY', '1').strip().lower() in ('1', 'true', 'yes', 'on')
    tracker.tracker.multi_frame_config.l25_motion_reliability_mode = os.getenv('L25_MOTION_RELIABILITY_MODE', 'manual').strip().lower()
    tracker.tracker.multi_frame_config.l25_motion_reliability_model_path = os.getenv('L25_MOTION_RELIABILITY_MODEL', '').strip()
    tracker.tracker.multi_frame_config.l25_motion_reliability_bias = float(os.getenv('L25_MOTION_RELIABILITY_BIAS', '-0.1'))
    tracker.tracker.multi_frame_config.l25_motion_reliability_score_gain = float(os.getenv('L25_MOTION_RELIABILITY_SCORE_GAIN', '2.0'))
    tracker.tracker.multi_frame_config.l25_motion_reliability_uncertainty_gain = float(os.getenv('L25_MOTION_RELIABILITY_UNCERTAINTY_GAIN', '2.0'))
    tracker.tracker.multi_frame_config.l25_velocity_share_min = float(os.getenv('L25_VEL_SHARE_MIN', '0.05'))
    tracker.tracker.multi_frame_config.l25_velocity_share_max = float(os.getenv('L25_VEL_SHARE_MAX', '0.30'))
    tracker.tracker.multi_frame_config.l25_velocity_geom_center = float(os.getenv('L25_VEL_GEOM_CENTER', '0.30'))
    tracker.tracker.multi_frame_config.l25_velocity_geom_gain = float(os.getenv('L25_VEL_GEOM_GAIN', '10.0'))
    tracker.tracker.multi_frame_config.l25_velocity_motion_center = float(os.getenv('L25_VEL_MOTION_CENTER', '0.65'))
    tracker.tracker.multi_frame_config.l25_velocity_motion_gain = float(os.getenv('L25_VEL_MOTION_GAIN', '8.0'))
    tracker.tracker.multi_frame_config.l25_velocity_reliability_strength = float(os.getenv('L25_VEL_RELIABILITY_STRENGTH', '0.60'))
    tracker.tracker.multi_frame_config.verbose = False  # 关闭调试输出
    tracker.tracker.multi_frame_config.use_l25_memory_bank_appearance = False
    tracker.tracker.multi_frame_config.geometry_mode = l25_geometry_mode
    tracker.tracker.multi_frame_config.use_rotated_geom_in_l25 = False
    tracker.tracker.multi_frame_config.rotated_geom_weight_l25 = 0.10
    tracker.tracker.multi_frame_config.memory_bank_size = 3
    tracker.tracker.multi_frame_config.memory_bank_min_conf = 0.4
    tracker.tracker.multi_frame_config.memory_bank_rescore_margin = 0.03
    tracker.tracker.multi_frame_config.enable_candidate_pre_gate = False
    tracker.tracker.multi_frame_config.candidate_min_iou = 0.03
    tracker.tracker.multi_frame_config.candidate_min_size_ratio = 0.55
    tracker.tracker.multi_frame_config.candidate_max_center_dist_base = 2.0
    tracker.tracker.multi_frame_config.candidate_max_center_dist_per_dt = 0.35
    tracker.tracker.multi_frame_config.enable_candidate_diag_log = os.getenv('L25_CANDIDATE_DIAG', '1').strip().lower() in ('1', 'true', 'yes', 'on')
    tracker.tracker.multi_frame_config.candidate_diag_log_path = os.path.join('logs', 'l25_candidate_diag.log')
    print(
        '[L2.5 Geometry] mode={} uncertainty_norm={} vel_conf_floor={}'.format(
            tracker.tracker.multi_frame_config.geometry_mode,
            tracker.tracker.multi_frame_config.uncertainty_norm,
            tracker.tracker.multi_frame_config.velocity_confidence_floor,
        )
    )
    print(
        '[L2.5 MotionRedistribute] reliability={} mode={} vel_share_min={} vel_share_max={} geom_center={} motion_center={}'.format(
            tracker.tracker.multi_frame_config.enable_confidence_aware_motion_l25,
            tracker.tracker.multi_frame_config.l25_motion_reliability_mode,
            tracker.tracker.multi_frame_config.l25_velocity_share_min,
            tracker.tracker.multi_frame_config.l25_velocity_share_max,
            tracker.tracker.multi_frame_config.l25_velocity_geom_center,
            tracker.tracker.multi_frame_config.l25_velocity_motion_center,
        )
    )
    print(
        '[Recovery Branch] enabled={} l15={} l25={}'.format(
            enable_recovery_branch,
            tracker.tracker.velocity_backtrack_enabled,
            tracker.tracker.multi_frame_config.enable_multi_frame_backtrack,
        )
    )
    tracker.tracker.enable_l4_identity_takeover = enable_l4_takeover
    tracker.tracker.enable_l4_identity_tempering = True
    tracker.tracker.enable_l4_handover_diag_log = os.getenv('L4_HANDOVER_DIAG', '0').strip().lower() in ('1', 'true', 'yes', 'on')
    tracker.tracker.l4_handover_diag_log_path = os.path.join('logs', 'l4_handover_diag.log')
    tracker.tracker.l4_handover_hits_center = float(
        os.getenv('L4_HANDOVER_HITS_CENTER', str(tracker.tracker.l4_handover_hits_center))
    )
    if not enable_l4_takeover:
        tracker.tracker.l4_handover_score_threshold = 2.0
    else:
        tracker.tracker.l4_handover_score_threshold = float(
            os.getenv('L4_HANDOVER_SCORE_THRESHOLD', str(tracker.tracker.l4_handover_score_threshold))
        )
    print(
        '[L4 Handover] takeover={} tempered={} hits_center={} age_center={} score_threshold={}'.format(
            tracker.tracker.enable_l4_identity_takeover,
            tracker.tracker.enable_l4_identity_tempering,
            tracker.tracker.l4_handover_hits_center,
            tracker.tracker.l4_handover_age_center,
            tracker.tracker.l4_handover_score_threshold,
        )
    )
    
    # ========== 基线：关闭加速度门控 ==========
    tracker.tracker.multi_frame_config.enable_l25_cooldown = False
    tracker.tracker.multi_frame_config.l25_cooldown_frames = 8
    tracker.tracker.multi_frame_config.allowed_backtrack_dts = {1, 2, 4, 5}
    tracker.tracker.multi_frame_config.use_acceleration_gate = False
    tracker.tracker.multi_frame_config.acceleration_threshold = 1.5  # 加速度阈值 (m/s²)
    
    # ========== 其他配置 ==========
    tracker.tracker.embedding_off = False  # 保持外观特征提取
    if tracker.tracker.enable_l15_motion_diag_log:
        open(tracker.tracker.l15_motion_diag_log_path, 'w', encoding='utf-8').close()
    if tracker.tracker.enable_l12_defer_diag_log:
        open(tracker.tracker.l12_defer_diag_log_path, 'w', encoding='utf-8').close()
    if tracker.tracker.multi_frame_config.enable_candidate_diag_log:
        open(tracker.tracker.multi_frame_config.candidate_diag_log_path, 'w', encoding='utf-8').close()
    if tracker.tracker.enable_l4_handover_diag_log:
        open(tracker.tracker.l4_handover_diag_log_path, 'w', encoding='utf-8').close()
    
    # Iterate through each data set 遍历数据集
    seq_filter = get_env_list('SEQ_FILTER')
    max_frame_override = get_env_int('MAX_FRAME', None)

    for seq_file_3D in detection_file_list_3D:
        seq_filename_txt, seq_id, _ = fileparts(seq_file_3D)
        if seq_filter is not None and seq_id not in seq_filter:
            continue
        tracker.tracker.multi_frame_config.current_seq_id = seq_id
        gt_by_frame = {}
        if 'train' in parts or 'training' in parts:
            gt_by_frame = load_kitti_tracking_gt_by_frame(
                os.path.join(data_root, 'label_02', f'{seq_id}.txt'),
                allowed_classes={'Car', 'Van'},
            )
        print('--------------Start processing the {} dataset--------------'.format(seq_id))
        total_image = 0  # Record the total frames in this dataset记录此数据集的总帧数
        # Find matching 2D detection file by sequence id
        seq_file_2D = None
        for f2d in detection_file_list_2D:
            s2d_filename_txt, s2d_id, _ = fileparts(f2d)
            if s2d_id == seq_id:
                seq_file_2D = f2d
                break
        if seq_file_2D is None:
            print(f"⚠️  序列 {seq_id} 缺少2D检测，写入空结果")
            i += 1
            continue
        txt_path = txt_path_0 + "\\" + seq_id + '.txt'
        image_path = image_path_0 + '\\' + seq_id; mkdir_if_inexistence(image_path)
        assoc_diag_path = assoc_diag_path_0 + "\\" + seq_id + '.csv'
        open(txt_path, 'w').close()
        with open(assoc_diag_path, 'w', encoding='utf-8') as f_assoc:
            f_assoc.write('frame,track_id,assoc_level,assoc_frame,time_since_update,hits\n')
        tracker.tracker.velocity_weight_vmax = 10.0
        tracker.tracker.adaptive_threshold_low = 0.55
        tracker.tracker.adaptive_threshold_mid = 0.60
        tracker.tracker.adaptive_threshold_high = 0.70

        calib_file = [calib_file for calib_file in calib_files if calib_file==seq_filename_txt]
        calib_file_seq = os.path.join(calib_root, ''.join(calib_file))
        image_dir = os.path.join(dataset_dir, seq_id)
        #image_dir = dataset_dir
        image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        seq_dets_3D = np.loadtxt(seq_file_3D, delimiter=',')  # load 3D detections, N x 15
        seq_dets_2D = np.loadtxt(seq_file_2D, delimiter=',')  # load 2D detections, N x 6
        
        # 跳过空文件
        if seq_dets_3D.size == 0 or seq_dets_2D.size == 0:
            print(f"⚠️  序列 {seq_id} 检测为空，跳过")
            i += 1
            continue
        
        # 确保是 2D 数组
        if seq_dets_3D.ndim == 1:
            seq_dets_3D = seq_dets_3D.reshape(1, -1)
        if seq_dets_2D.ndim == 1:
            seq_dets_2D = seq_dets_2D.reshape(1, -1)

        min_frame, max_frame = int(seq_dets_3D[:, 0].min()), len(image_filenames)
        if max_frame_override is not None:
            max_frame = min(max_frame, int(max_frame_override))

        for frame, img0_path in zip(range(min_frame, max_frame + 1), image_filenames):
            tracker.tracker.multi_frame_config.current_data_frame = frame
            tracker.current_gt_frame_entries = gt_by_frame.get(frame, [])
            img_0 = cv2.imread(img0_path)
            _, img0_name, _ = fileparts(img0_path)
            dets_3D_camera = seq_dets_3D[seq_dets_3D[:, 0] == frame, 7:14]  # 3D bounding box(h,w,l,x,y,z,theta)
            dets_8corners = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets_3D_camera]

            ori_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, -1].reshape((-1, 1))
            other_array = seq_dets_3D[seq_dets_3D[:, 0] == frame, 1:7]
            additional_info = np.concatenate((ori_array, other_array), axis=1) #concatenate拼接函数

            dets_3Dto2D_image = seq_dets_3D[seq_dets_3D[:, 0] == frame, 2:6]
            if len(seq_dets_2D)!=0:
                try:
                    dets_2D = seq_dets_2D[seq_dets_2D[:, 0] == frame, 1:5]   # 2D bounding box(x1,y1,x2,y2)
                    dets_2D_conf = seq_dets_2D[seq_dets_2D[:, 0] == frame, -1]
                except:
                    print(seq_dets_2D)
            else:
                dets_2D = []
                dets_2D_conf = []

            # Data Fusion(3D and 2D detections)
            detection_2D_fusion, detection_3Dto2D_fusion, detection_3D_fusion, detection_2D_only, detection_3Dto2D_only, detection_3D_only,detection_2D_only_conf,detection_3D_fusion_conf = \
                datafusion2Dand3D(dets_3D_camera, dets_2D, dets_3Dto2D_image, additional_info,dets_2D_conf)

            detection_2D_only_tlwh = np.array([convert_x1y1x2y2_to_tlwh(i) for i in detection_2D_only]) # (x1,y1,x2,y2) to (x,y,center_x,center_y)

            start_time = time.time()
            trackers,trackers_2d,trackers_meta = tracker.update(detection_3D_fusion, detection_2D_only_tlwh, detection_3D_only, detection_3Dto2D_only,
                                      additional_info, calib_file_seq,img_0, detection_2D_only_conf, detection_3D_fusion_conf)
            cycle_time = time.time() - start_time
            total_time += cycle_time

            # Outputs
            total_frames += 1 # Total frames for all datasets
            total_image += 1 # Total frames for a dataset
            if total_image % 50 == 0:
                print("Now start processing the {} image of the {} dataset".format(total_image, seq_id))

            img_vis = img_0.copy()
            if len(trackers) > 0:
                for meta_idx, d in enumerate(trackers):
                    bbox3d = d.flatten()
                    bbox3d_tmp = bbox3d[1:8]  # 3D bounding box(h,w,l,x,y,z,theta)
                    id_tmp = int(bbox3d[0])
                    ori_tmp = bbox3d[8]
                    type_tmp = det_id2str[bbox3d[9]]
                    bbox2d_tmp_trk = bbox3d[10:14]
                    conf_tmp = bbox3d[14]
                    color = compute_color_for_id(id_tmp)
                    label = f'{id_tmp} {type_tmp}'
                    image_save_path = os.path.join(image_path, '%06d.jpg' % (int(img0_name)))
                    with open(txt_path, 'a') as f:
                        str_to_srite = (
                            f"{frame:d} {id_tmp:d} {type_tmp} 0 0 "
                            f"{ori_tmp:.6f} "
                            f"{bbox2d_tmp_trk[0]:.6f} {bbox2d_tmp_trk[1]:.6f} {bbox2d_tmp_trk[2]:.6f} {bbox2d_tmp_trk[3]:.6f} "
                            f"{bbox3d_tmp[0]:.6f} {bbox3d_tmp[1]:.6f} {bbox3d_tmp[2]:.6f} {bbox3d_tmp[3]:.6f} "
                            f"{bbox3d_tmp[4]:.6f} {bbox3d_tmp[5]:.6f} {bbox3d_tmp[6]:.6f} "
                            f"{conf_tmp:.6f}\n"
                        )
                        f.write(str_to_srite)
                    if meta_idx < len(trackers_meta):
                        meta = trackers_meta[meta_idx]
                        with open(assoc_diag_path, 'a', encoding='utf-8') as f_assoc:
                            f_assoc.write(
                                '{},{},{},{},{},{}\n'.format(
                                    int(frame),
                                    int(meta.get('track_id', id_tmp)),
                                    str(meta.get('assoc_level', 'UNKNOWN')),
                                    int(meta.get('assoc_frame', -1)),
                                    int(meta.get('time_since_update', -1)),
                                    int(meta.get('hits', -1)),
                                )
                            )
                        #show_image_with_boxes(img_vis, bbox3d_tmp, image_path, color, img0_name, label, calib_file_seq,line_thickness=1)  # 禁用可视化（LiDAR坐标）
                        show_image_with_boxes(img_vis, bbox3d_tmp, image_path, color, img0_name, label, calib_file_seq, line_thickness=1)
                        plot_one_box(
                            np.array([id_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3]]),
                            img_vis,
                            image_path,
                            color,
                            img0_name,
                            label,
                            line_thickness=1,
                        )
            #if len(trackers_2d) > 0:
                #for d in trackers_2d:
                    #bbox2d = d.flatten()
                    # print(bbox2d,type(bbox2d))
                    #bbox2d_tmp = bbox2d[1:5]
                    #id_tmp = int(bbox2d[0])
                    #color = compute_color_for_id(id_tmp)
                    #image_save_path = os.path.join(image_path, '%06d.jpg' % (int(img0_name)))
                    #label = f'{id_tmp} {"car"}'
                    #with open(txt_path, 'a') as f:
                        #type_tmp = 'car'
                        #str_to_srite = (
                            #f"{frame:d} {id_tmp:d} {type_tmp} -1 -1 -10 "
                            #f"{bbox2d_tmp[0]:.6f} {bbox2d_tmp[1]:.6f} {bbox2d_tmp[2]:.6f} {bbox2d_tmp[3]:.6f} "
                            #f"-1000 -1000 -1000 -10 -1 -1 -1 -1\n"
                        #)
                        #f.write(str_to_srite)
                        # plot_one_box(bbox2d,img_0,image_path,color,img0_name,label,line_thickness=1)
                        # print(image_save_path)

        i += 1
        print('--------------The time it takes to process all datasets are {}s --------------'.format(total_time))
    
    # 输出 L1.5 和 L2.5 恢复统计
    print('============== 轨迹恢复统计 ==============')
    print(f'L1.5 (速度回溯) 总恢复: {tracker.tracker.total_L15_recoveries}')
    print(f'L2.5 (多帧回溯) 总恢复: {tracker.tracker.total_L25_recoveries}')
    print(f'L1 defer 总延迟对数: {tracker.tracker.total_l12_deferred_pairs}')
    print(f'L1 defer 后被 L1.5 接回: {tracker.tracker.total_l12_deferred_recovered_l15}')
    print(f'L4 总匹配: {tracker.tracker.total_l4_matches}')
    print(f'L4 身份接管: {tracker.tracker.total_l4_id_takeovers}')
    print(f'L4 仅救回不接管: {tracker.tracker.total_l4_id_kept}')
    print('===========================================')
    
    # Per-level timing summary
    t1 = tracker.tracker.t_L1
    t15 = tracker.tracker.t_L15
    t2 = tracker.tracker.t_L2
    t3 = tracker.tracker.t_L3
    t4 = tracker.tracker.t_L4
    t_sum = t1 + t15 + t2 + t3 + t4
    t_other = max(total_time - t_sum, 0.0)
    def pct(x):
        return 100.0 * x / total_time if total_time > 0 else 0.0
    print('============== Per-Level Time ==============')
    print('L1   (融合3D):      {:.3f}s ({:.2f}%)'.format(t1, pct(t1)))
    print('L1.5 (速度回溯):    {:.3f}s ({:.2f}%)'.format(t15, pct(t15)))
    print('L2   (仅3D):        {:.3f}s ({:.2f}%)'.format(t2, pct(t2)))
    print('L3   (仅2D):        {:.3f}s ({:.2f}%)'.format(t3, pct(t3)))
    print('L4   (2D→3D跨域):  {:.3f}s ({:.2f}%)'.format(t4, pct(t4)))
    print('其他(数据装载/写盘等): {:.3f}s ({:.2f}%)'.format(t_other, pct(t_other)))
    print('===========================================')
    print('--------------FPS = {} --------------'.format(total_frames/total_time))


if __name__ == '__main__':
    main()


