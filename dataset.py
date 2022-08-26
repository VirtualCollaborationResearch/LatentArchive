import os
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoFeatureExtractor

extractor = AutoFeatureExtractor.from_pretrained("Intel/dpt-large")

class CustomImageDataset(Dataset):
    def __init__(self, img, depth, transform=None, histogram_matching=None):
        self.depth = depth
        self.img = img
        self.transform = transform
        self.histogram_matching = histogram_matching

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        
        if self.transform:
            image = self.transform(self.img[idx])
        if self.histogram_matching:
            image = self.histogram_matching(image)

        item = extractor(images=image, return_tensors="pt")

        item['labels'] =  self.depth[idx]
        return item