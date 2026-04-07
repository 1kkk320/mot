import torch

from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReIDVeRi(torch.nn.Module):
    """
    FastReID适配器 - 专门用于VeRi车辆ReID模型
    使用SBS(R50-ibn)架构，在VeRi数据集上预训练
    性能: Rank@1 97.0%, mAP 81.9%
    """
    def __init__(self, weights_path="external/weights/veri_sbs_R50-ibn.pth"):
        super().__init__()
        # 使用VeRi专用配置文件
        config_file = "external/fast_reid/configs/VeRi/sbs_R50-ibn.yml"
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()

        # 加载VeRi预训练权重
        Checkpointer(self.model).load(weights_path)
        
        # 转换为半精度以匹配输入
        self.model.half()
        
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST
        
        print(f"✅ VeRi ReID模型加载成功")
        print(f"   配置文件: {config_file}")
        print(f"   权重文件: {weights_path}")
        print(f"   输入尺寸: {self.pW}x{self.pH}")
        print(f"   预期性能: Rank@1 97.0%, mAP 81.9%")

    def forward(self, batch):
        """
        前向传播
        Args:
            batch: 输入图像batch，shape为 [N, C, H, W]
        Returns:
            features: 特征向量，shape为 [N, feature_dim]
        """
        # 使用半精度加速推理
        batch = batch.half()
        with torch.no_grad():
            features = self.model(batch)
        return features