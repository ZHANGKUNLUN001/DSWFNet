import os
import random
import sys
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
# from altair import Parameter
from .SyncTransform import DeepSyncTransform
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Parameter import SET_NAME

class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, work_mode=0, shuffle=True,
                 set_name='DeepGlobe_Road_Extraction_Dataset'):
        super().__init__()

        self.image_dir = image_dir
        self.label_dir = image_dir
        self.work_mode = work_mode
        self.shuffle = shuffle
        self.set_name = set_name

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.tiff'))]
        self.label_files = []

        for image_file in self.image_files:
            image_id, ext = os.path.splitext(image_file)
            if ext == ".jpg" and image_id.endswith("_sat"):
                image_id = image_id[:-4]
                label_file = f"{image_id}_mask.png"
            elif ext == ".tiff":
                label_file = f"{image_id}.tif"
            else:
                continue
            self.label_files.append(label_file)

        assert len(self.image_files) == len(self.label_files), "图像与标签数量不匹配"

        if self.shuffle:
            combined = list(zip(self.image_files, self.label_files))
            random.shuffle(combined)
            self.image_files, self.label_files = zip(*combined)

        self.transform_sat = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_road = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x >= 0.5).float())
        ])

        self.transform_sat_mas = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_road_mas = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x >= 0.5).float())
        ])


        # 增强策略
        self.sync_transform = DeepSyncTransform(crop_size=512) if self.work_mode == 1 else None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.sync_transform:
            image, label = self.sync_transform(image, label)
            image = self.transform_sat(image)
            label = self.transform_road(label)
        else:
            # ✅ 验证阶段，根据数据集切换 transform
            if SET_NAME == 'Massachusetts_Roads_Dataset':
                image = self.transform_sat_mas(image)
                label = self.transform_road_mas(label)
            elif SET_NAME == 'CHN6-CUG':
                image = self.transform_sat(image)
                label = self.transform_road(label)
        return image, label