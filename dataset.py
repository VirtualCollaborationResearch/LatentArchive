import os
import pandas as pd
from torchvision.io import read_image
import cv2 as cv
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, depth_dir, transform=None, target_transform=None):
        self.depth_dir = depth_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_path = os.path(self.img_dir + '/' + os.listdir(self.img_dir)[idx])
        depth_path = os.path(self.depth_dir + '/' + os.listdir(self.depth_dir)[idx])
        image = cv.imread(img_path, cv.IMREAD_ANYDEPTH)
        depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH)
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        return image, depth