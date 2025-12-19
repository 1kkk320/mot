"""
KITTI跟踪评估 - MOTA和ID Switch指标
评估CLEAR指标族: MOTA, MOTP, ID Switch, FP, FN等
"""
import sys
import os

# 添加TrackEval到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "external/TrackEval"))
import trackeval

if __name__ == "__main__":
    # 评估配置
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['USE_PARALLEL'] = False
    eval_config['PRINT_RESULTS'] = True
    eval_config['PRINT_ONLY_COMBINED'] = False
    eval_config['OUTPUT_SUMMARY'] = True
    eval_config['OUTPUT_DETAILED'] = True
    eval_config['PLOT_CURVES'] = False
    eval_config['PRINT_CONFIG'] = False
    
    # 数据集配置
    dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = r'e:\mot\datasets\kitti\train'
    dataset_config['TRACKERS_FOLDER'] = r'e:\mot\results'
    dataset_config['TRACKERS_TO_EVAL'] = ['virconv_OCM']
    dataset_config['CLASSES_TO_EVAL'] = ['car']  # 只评估car,因为没有检测到pedestrian
    dataset_config['SPLIT_TO_EVAL'] = 'training'
    dataset_config['TRACKER_SUB_FOLDER'] = 'data'
    dataset_config['OUTPUT_SUB_FOLDER'] = ''
    dataset_config['PRINT_CONFIG'] = False
    
    # 指标配置（用于 CLEAR/Identity）。HOTA 使用默认配置。
    metrics_config = {'THRESHOLD': 0.5, 'PRINT_CONFIG': False}
    
    print("=" * 80)
    print("KITTI 跟踪评估 - MOTA & ID Switch")
    print("=" * 80)
    print(f"跟踪器: virconv_OCM")
    print(f"评估类别: {dataset_config['CLASSES_TO_EVAL']}")
    print(f"IoU阈值: {metrics_config['THRESHOLD']}")
    print("=" * 80)
    
    # 创建评估器
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
    # 可选：通过环境变量 SEQ_WHITELIST 控制评估序列（与原逻辑保持一致）
    seq_env = os.environ.get('SEQ_WHITELIST', '').strip()
    if seq_env:
        seqs = [s.strip() for s in seq_env.split(',') if s.strip()]
        ds = dataset_list[0]
        if hasattr(ds, 'seq_list'):
            ds.seq_list = seqs
        for attr in ('seq_list_all', 'seqs', 'seq_list_full'):
            if hasattr(ds, attr):
                setattr(ds, attr, seqs)
    
    # 指标列表：HOTA（含 HOTA/DetA/AssA/LocA）、CLEAR、Identity（IDF1 等）
    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity()
    ]

    print("\n正在评估指标...")
    print("包含指标:")
    print("  - HOTA 系列: HOTA, DetA, AssA, LocA 等")
    print("  - CLEAR 系列: MOTA, MOTP, MT, ML, FP, FN, IDSW, Frag")
    print("  - Identity 系列: IDF1, IDP, IDR")
    print("-" * 80)
    
    evaluator.evaluate(dataset_list, metrics_list)
    
    print("\n" + "=" * 80)
    print("HOTA / CLEAR / Identity 评估完成！")
    print("=" * 80)
    print("\n关键指标说明:")
    print("  MOTA: 多目标跟踪准确度 (-∞到100, 越高越好)")
    print("    计算公式: MOTA = 1 - (FN + FP + ID Switch) / GT")
    print("  MOTP: 多目标跟踪精度 (0-100, 越高越好)")
    print("    表示检测框与GT的平均IoU")
    print("  ID Switch: 身份切换次数 (越少越好)")
    print("    表示同一目标被分配不同ID的次数")
    print("  MT: 大部分时间被跟踪的目标数")
    print("  ML: 大部分时间丢失的目标数")
    print("=" * 80)
