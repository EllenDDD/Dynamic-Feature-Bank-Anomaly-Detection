# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random
import hashlib
import copy
import cv2
from .build import DATASET_REGISTRY
import logging

from PIL import Image
import numpy as np

from torchvision import utils as vutils

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform

logging.getLogger('PIL').setLevel(logging.WARNING)

def tile_image(img, stride_ratio=0.8):
    height, width, _ = img.shape
    shorter_edge = min(height, width)
    tile_size = shorter_edge
    stride = int(tile_size * stride_ratio)

    tile_image = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract a tile from the image
            tile = img[y:y + tile_size, x:x + tile_size]
            tile_image.append(tile)

    return tile_image


@DATASET_REGISTRY.register()
class IC_dataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        normal_json_path_list: list,
        outlier_json_path_list: list,
        transform: Optional[Callable] = None,
        shot = None,
        reference_normal_json_path=None,
        fixed_reference_seed=None,
    ) -> None:
        super().__init__(root)

        self.normal_samples = []
        self.outlier_samples = []
        self.reference_normal_samples = []
        self.image = []
        self.total_n = 0
        self.total_o = 0
        self.shot = shot

        if isinstance(normal_json_path_list, str):
            normal_json_path_list = [normal_json_path_list]

        if isinstance(outlier_json_path_list, str):
            outlier_json_path_list = [outlier_json_path_list]

        normal_json_path_list = [p.replace("\\", "/") for p in normal_json_path_list]
        outlier_json_path_list = [p.replace("\\", "/") for p in outlier_json_path_list]

        for json_path in normal_json_path_list:
            with open(json_path, "r", encoding="utf-8") as handle:
                self.normal_samples.extend(json.load(handle))

        for json_path in outlier_json_path_list:
            with open(json_path, "r", encoding="utf-8") as handle:
                self.outlier_samples.extend(json.load(handle))

        if reference_normal_json_path is None:
            self.reference_normal_samples = self.normal_samples
        else:
            if isinstance(reference_normal_json_path, str):
                reference_normal_json_path = [reference_normal_json_path]
            for json_path in reference_normal_json_path:
                with open(json_path.replace("\\", "/"), "r", encoding="utf-8") as handle:
                    self.reference_normal_samples.extend(json.load(handle))

        if not self.reference_normal_samples:
            raise ValueError("The normal reference pool is empty.")
        invalid_references = [
            sample for sample in self.reference_normal_samples if int(sample.get("target", -1)) != 0
        ]
        if invalid_references:
            raise ValueError("The normal reference pool contains anomaly-labelled records.")

        self.transform = transform
        self.total_n, self.total_o = len(self.normal_samples), len(self.outlier_samples)
        self.image = self.normal_samples + self.outlier_samples
        self.fixed_reference_samples = None
        if fixed_reference_seed is not None:
            self.fixed_reference_samples = {}
            categories = sorted({sample["type"] for sample in self.image})
            for category in categories:
                candidates = [
                    sample for sample in self.reference_normal_samples
                    if sample["type"] == category
                ]
                if len(candidates) < self.shot:
                    raise ValueError(
                        "Category '{}' has {} clean reference images, but shot={}.".format(
                            category, len(candidates), self.shot
                        )
                    )
                seed_bytes = "{}:{}".format(fixed_reference_seed, category).encode("utf-8")
                category_seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:8], "big")
                self.fixed_reference_samples[category] = random.Random(category_seed).sample(
                    candidates, self.shot
                )

    def _load_image(self, path: str) -> Image.Image:
        if 'npy' in path[-3:]:
            img = np.load(path)
            return img
        return Image.open(path)    #.convert('RGB')

    def _combine_images(self, image, image2):
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.image[index]
        image = self._load_image(sample['image_path'])
        label = sample['target']
        sample_type = sample['type']

        # decide mode for interpolation
        cur_transforms = self.transform
        image = cur_transforms(image)

        # sample the normal samples belonging to the same type
        if self.fixed_reference_samples is not None:
            normal_index = self.fixed_reference_samples[sample_type]
        else:
            sample_path = os.path.normcase(os.path.realpath(sample["image_path"]))
            same_normal_samples = [
                candidate for candidate in self.reference_normal_samples
                if candidate['type'] == sample_type
                and (
                    int(label) != 0
                    or os.path.normcase(os.path.realpath(candidate["image_path"])) != sample_path
                )
            ]
            if len(same_normal_samples) < self.shot:
                raise ValueError(
                    "Category '{}' has {} independent clean reference images, but shot={}.".format(
                        sample_type, len(same_normal_samples), self.shot
                    )
                )
            normal_index = random.sample(same_normal_samples, self.shot)

        image_list = list()
        image_list.append(image)
        for i in range(len(normal_index)):
            assert normal_index[i]['type'] == sample_type
            n_img = self._load_image(normal_index[i]['image_path'])
            n_img = cur_transforms(n_img)
            image_list.append(n_img)

        image_type = sample_type

        return image_list, image_type, label

    def __len__(self) -> int:
        return len(self.image)
