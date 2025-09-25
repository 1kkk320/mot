import pdb
from collections import OrderedDict
import os
import pickle

import torch
import cv2
import torchvision
import torchreid
import numpy as np
from thop import profile
from torchstat import stat
from external.adaptors.fastreid_adaptor import FastReID
from pathlib import Path


class EmbeddingComputer:
    def __init__(self, grid_off):
        self.model = None
        self.crop_size = (128, 384)
        os.makedirs("./cache/embeddings/", exist_ok=True)
        self.cache_path = "./cache/embeddings/{}_embedding.pkl"
        self.cache = {}
        self.cache_name = ""
        self.grid_off = grid_off
        self.normalize = False
        self.max_batch = 1024
        self.flops = 16.446
        self.total_param = 0

    def get_param_flop(self):
        model = self.model
        inputs = torch.randn(1,3,112,112)
        self.flops,self.total_param = profile(model,(inputs,))
        stat(model,(3,224,224))
        print("flops:",self.flops,"params:",self.total_param)

    def get_param_flop2(self):
        model = self.model
        for param in model.parameters():
            mulvalue = np.prod(param.size())
            self.total_param +=mulvalue
        # model.cuda()
        # stat(model, (3, 224, 224))

    def load_cache(self, path):
        self.cache_name = path
        cache_path = self.cache_path.format(path)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.cache = pickle.load(fp)

    def compute_embedding(self, img, bbox, is_numpy=True):

        # if self.model is None:
        #     self.initialize_model()
        #
        # # Make sure bbox is within image frame
        # if is_numpy:
        #     h, w = img.shape[:2]
        # else:
        #     h, w = img.shape[2:]
        # results = np.round(bbox).astype(np.int32)
        # results[:, 0] = results[:, 0].clip(0, w)
        # results[:, 1] = results[:, 1].clip(0, h)
        # results[:, 2] = results[:, 2].clip(0, w)
        # results[:, 3] = results[:, 3].clip(0, h)
        #
        # # Generate all the crops
        # crops = []
        # for p in results:
        #     if is_numpy:
        #         crop = img[p[1] : p[3], p[0] : p[2]]
        #         try:
        #             crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        #             crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
        #             crop = torch.as_tensor(crop.astype("float32").transpose(2, 0, 1))
        #             crop = crop.unsqueeze(0)
        #         except:
        #             if len(p) == 0:
        #                 print(crop,type(crop))
        #                 pass
        #
        #     else:
        #         crop = img[:, :, p[1] : p[3], p[0] : p[2]]
        #         crop = torchvision.transforms.functional.resize(crop, self.crop_size)
        #
        #     crops.append(crop)
        #
        # crops = torch.cat(crops, dim=0)
        #
        # # Create embeddings and l2 normalize them
        # with torch.no_grad():
        #     crops = crops.cuda()
        #     crops = crops.half()
        #     embs = self.model(crops)
        # embs = torch.nn.functional.normalize(embs)
        # embs = embs.cpu().numpy()
        #
        # # self.cache[tag] = embs
        # return embs
        #
        #
        bbox = np.asarray(bbox)
        if self.model is None:
            self.initialize_model()

        # Generate all of the patches
        crops = []
        if self.grid_off:
            # Basic embeddings
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
                if self.normalize:
                    crop /= 255
                    crop -= np.array((0.485, 0.456, 0.406))
                    crop /= np.array((0.229, 0.224, 0.225))
                crop = torch.as_tensor(crop.transpose(2, 0, 1))
                crop = crop.unsqueeze(0)
                crops.append(crop)
        else:
            # Grid patch embeddings
            for idx, box in enumerate(bbox):
                crop = self.get_horizontal_split_patches(img, box)
                crops.append(crop)
        crops = torch.cat(crops, dim=0)

        # Create embeddings and l2 normalize them
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
        # print("flops:", self.flops, "params:", self.total_param/1e6)


        return embs

    def initialize_model(self):
        """
        model = torchreid.models.build_model(name="osnet_ain_x1_0", num_classes=2510, loss="softmax", pretrained=False)
        sd = torch.load("external/weights/osnet_ain_ms_d_c.pth.tar")["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in sd.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()
        model.cuda()
        """
        # if self.dataset == "mot17":
        #     path = "external/weights/mot17_sbs_S50.pth"
        # elif self.dataset == "mot20":
        #     path = "external/weights/mot20_sbs_S50.pth"
        # elif self.dataset == "dance":
        #     path = None
        # else:
        #     raise RuntimeError("Need the path for a new ReID model.")
        path = "external/weights/market_sbs_S50.pth"
        model = FastReID(path)
        model.eval()
        model.cuda()
        model.half()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.model = model
        self.get_param_flop2()

    def dump_cache(self):
        if self.cache_name:
            with open(self.cache_path.format(self.cache_name), "wb") as fp:
                pickle.dump(self.cache, fp)

    def get_horizontal_split_patches(self, image, bbox):
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            h, w = image.shape[2:]

        bbox = np.array(bbox)
        bbox = bbox.astype(np.int)
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > w or bbox[3] > h:
            # Faulty Patch Correction
            bbox[0] = np.clip(bbox[0], 0, None)
            bbox[1] = np.clip(bbox[1], 0, None)
            bbox[2] = np.clip(bbox[2], 0, image.shape[1])
            bbox[3] = np.clip(bbox[3], 0, image.shape[0])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        ### TODO - Write a generalized split logic
        split_boxes = [
            [x1, y1, x1 + w, y1 + h / 3],
            [x1, y1 + h / 3, x1 + w, y1 + (2 / 3) * h],
            [x1, y1 + (2 / 3) * h, x1 + w, y1 + h],
        ]

        split_boxes = np.array(split_boxes, dtype="int")
        patches = []
        # breakpoint()
        for ix, patch_coords in enumerate(split_boxes):
            if isinstance(image, np.ndarray):
                im1 = image[patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2], :]

                patch = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, self.crop_size, interpolation=cv2.INTER_LINEAR)
                patch = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
                patch = patch.unsqueeze(0)
                # print("test ", patch.shape)
                patches.append(patch)
            else:
                im1 = image[:, :, patch_coords[1] : patch_coords[3], patch_coords[0] : patch_coords[2]]
                patch = torchvision.transforms.functional.resize(im1, (256, 128))
                patches.append(patch)

        patches = torch.cat(patches, dim=0)

        # print("Patches shape ", patches.shape)
        # patches = np.array(patches)
        # print("ALL SPLIT PATCHES SHAPE - ", patches.shape)

        return patches