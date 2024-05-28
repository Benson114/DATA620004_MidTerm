import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import random_split, Dataset

from src.config import *


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        """
        root_dir: 数据集目录路径
        is_train: 是否加载训练集
        transform: 应用于图像的预处理函数
        """
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip images vertically
            transforms.RandomRotation(45),  # Randomly rotate images within ±45 degrees
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_info = pd.read_csv(
            f"{root_dir}/images.txt",
            sep=r"\s+",
            names=["img_id", "img_name"]
        )
        self.label_info = pd.read_csv(
            f"{root_dir}/image_class_labels.txt",
            sep=r"\s+",
            names=["img_id", "label"]
        )
        self.split_info = pd.read_csv(
            f"{root_dir}/train_test_split.txt",
            sep=r"\s+",
            names=["img_id", "is_train"]
        )

        self.data_info = self.data_info.merge(self.label_info, on="img_id")
        self.data_info = self.data_info.merge(self.split_info, on="img_id")

        if is_train:
            self.data_info = self.data_info[self.data_info["is_train"] == 1]
        else:
            self.data_info = self.data_info[self.data_info["is_train"] == 0]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = f"{self.root_dir}/images/{self.data_info.iloc[idx]['img_name']}"
        image = Image.open(img_path).convert("RGB")
        label = self.data_info.iloc[idx]["label"] - 1
        image = self.transform(image)
        return image, label


train_valid_dataset = CUB200Dataset(root_dir=root_dir, is_train=True)
train_size = int((1 - dataloader_params["valid_rate"]) * len(train_valid_dataset))
valid_size = len(train_valid_dataset) - train_size
train_dataset, valid_dataset = random_split(train_valid_dataset, [train_size, valid_size])
test_dataset = CUB200Dataset(root_dir=root_dir, is_train=False)
