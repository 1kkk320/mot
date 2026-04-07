from collections import OrderedDict
from pathlib import Path
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np

from external.adaptors.fastreid_veri_adaptor import FastReIDVeRi


class EmbeddingComputerVeRi:
    """
    VeRi专用的EmbeddingComputer
    使用在VeRi数据集上预训练的SBS(R50-ibn)模型
    性能: Rank@1 97.0%, mAP 81.9%
    """
    def __init__(self, dataset, test_dataset, grid_off, max_batch=1024):
        self.model = None
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.crop_size = (128, 384)  # VeRi模型的输入尺寸
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_veri_embedding.pkl"  # 使用不同的缓存文件
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.max_batch = max_batch

        # VeRi模型不需要normalize（FastReID内部处理）
        self.normalize = False
        
        print("🚗 初始化VeRi车辆ReID EmbeddingComputer")
        print(f"   网格分割: {'关闭' if grid_off else '开启'}")
        print(f"   最大批次: {max_batch}")

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
            print(f"✅ 加载VeRi嵌入缓存: {cache_path}")

    def get_horizontal_split_patches(self, image, bbox, tag, idx, viz=False):
        """获取水平分割的图像块（用于网格特征）"""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            # 修正错误的边界框
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # 三等分水平分割（适合车辆：前部、中部、后部）
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2], :]

                if viz:
                    dirs = "./viz/{}/{}".format(tag.split(":")[0], tag.split(":")[1])
                    Path(dirs).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(dirs, "{}_{}.png".format(idx, ix)),
                        im1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255,
                    )
                    
                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, self.crop_size)
                patches.append(patch)

        patches = torch.cat(patches, dim=0)
        return patches

    def compute_embedding(self, img, bbox, tag):
        """计算车辆ReID嵌入特征"""
        if self.cache_name != tag.split(":")[0]:
            self.load_cache(tag.split(":")[0])

        # 确保 bbox 是 numpy 数组
        if isinstance(bbox, list):
            bbox = np.array(bbox)
        elif not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)

        if tag in self.cache:
            embs = self.cache[tag]
            if embs.shape[0] != bbox.shape[0]:
                raise RuntimeError(
                    "ERROR: 缓存的嵌入数量与检测数量不匹配。\n"
                    "检测器模型是否发生了变化？如果是，请删除缓存。"
                )
            return embs

        if self.model is None:
            self.initialize_model()

        # 处理空 bbox 情况
        if bbox.shape[0] == 0:
            return np.empty((0, 2048), dtype=np.float32)

        # 生成所有图像块
        crops = []
        if self.grid_off:
            # 基础嵌入（整个车辆）
            h, w = img.shape[:2]
            results = np.round(bbox).astype(np.int32)
            results[:, 0] = results[:, 0].clip(0, w)
            results[:, 1] = results[:, 1].clip(0, h)
            results[:, 2] = results[:, 2].clip(0, w)
            results[:, 3] = results[:, 3].clip(0, h)

            crops = []
            for p in results:
                crop = img[p[1] : p[3], p[0] : p[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
                # VeRi模型不需要手动normalize
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # 网格分割嵌入（车辆部件）
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box, tag, idx)
                crops.append(crop)
                
        crops = torch.cat(crops, dim=0)

        # 创建嵌入并L2归一化
        embs = []
        for idx in range(0, len(crops), self.max_batch):
            batch_crops = crops[idx : idx + self.max_batch]
            batch_crops = batch_crops.cuda()
            with torch.no_grad():
                batch_embs = self.model(batch_crops)
            embs.extend(batch_embs)
            
        embs = torch.stack(embs)
        embs = torch.nn.functional.normalize(embs, dim=-1)

        if not self.grid_off:
            embs = embs.reshape(bbox.shape[0], -1, embs.shape[-1])
        embs = embs.cpu().numpy()

        self.cache[tag] = embs
        return embs

    def initialize_model(self):
        """初始化VeRi预训练模型"""
        print("🚀 加载VeRi预训练模型...")
        
        # 使用VeRi专用模型
        model = FastReIDVeRi("external/weights/veri_sbs_R50-ibn.pth")
        model.eval()
        model.cuda()
        model.half()  # 使用半精度加速
        self.model = model
        
        print("✅ VeRi模型初始化完成")

    def dump_cache(self):
        """保存缓存"""
        if self.cache_name:
            cache_path = self.cache_path.format(self.cache_name)
            with open(cache_path, "wb") as fp:
                pickle.dump(self.cache, fp)
            print(f"💾 保存VeRi嵌入缓存: {cache_path}")