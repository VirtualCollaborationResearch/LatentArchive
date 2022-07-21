import os
import pandas as pd
from torchvision.io import read_image
import cv2 as cv
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, depth_dir,img_labels, transform=None, target_transform=None):
        self.depth_dir = depth_dir
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(img_labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv.imread(img_path, cv.IMREAD_REDUCED_COLOR_2 + cv.IMREAD_ANYDEPTH)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label