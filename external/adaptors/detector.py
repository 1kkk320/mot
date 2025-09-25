"""Generic detector."""
import os
import pickle

import torch

from external.adaptors import yolox_adaptor

'''
这个Detector类的主要功能是初始化对象检测模型，执行前向传播来检测物体，并将检测结果进行缓存以提高性能。
如果缓存中已经存在相同标签的检测结果，它将直接返回缓存结果，否则会计算新的检测结果并进行缓存。
这有助于加速对象检测的多次调用，特别是在处理大量数据时
'''
class Detector(torch.nn.Module):
    K_MODELS = {"yolox"}

    def __init__(self, model_type, path, dataset):
        super().__init__()
        if model_type not in self.K_MODELS:
            raise RuntimeError(f"{model_type} detector not supported")

        self.model_type = model_type
        self.path = path
        self.dataset = dataset
        self.model = None

        os.makedirs("./cache", exist_ok=True)
        self.cache_path = os.path.join(
            "./cache", f"det_{os.path.basename(path).split('.')[0]}.pkl"
        )
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as fp:
                self.cache = pickle.load(fp)
        else:
            self.initialize_model()

    def initialize_model(self):
        """Wait until needed."""
        if self.model_type == "yolox":
            self.model = yolox_adaptor.get_model(self.path, self.dataset)

    def forward(self, batch, tag=None):
        if tag in self.cache:
            return self.cache[tag]
        if self.model is None:
            self.initialize_model()

        with torch.no_grad():
            batch = batch.half()
            output = self.model(batch)
        if output is not None:
            self.cache[tag] = output.cpu()

        return output

    def dump_cache(self):
        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.cache, fp)
