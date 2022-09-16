# %%
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
import os

# %%
class AddGaussianNoise:
    """Add Gaussian noise to a tensor with a given mean and standard deviation"""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

# %%
def plot_histogram(img,title, bins = 100):
    img,grid = torch.histogram(img, bins=bins, density=True)
    img = img.numpy()
    grid = grid.numpy()
    
    plt.bar(grid[1:],img)
    plt.title(title)
    plt.show()

# %% [markdown]
# List of transforms to be applied

# %%
def rgb_transform(img):
    """Function that takes a tensor and returns it in gray scale and with additive noise"""
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.GaussianBlur(kernel_size=(7,13), sigma=(2, 4)),
                                    #transforms.RandomSolarize(threshold= 200, p=0.3),
                                    #transforms.Normalize((0.5), (0.5)), #May not be required after histogram matching
                                    AddGaussianNoise(mean=0.0, std=0.1)
                                    #transforms.ToPILImage()
                                    ])
    #img = img.transpose(0,2).transpose(1,2)
    img = transform(img)
    return img

def histogram_matching(img, ref_dir='./Manhatta Frames/'):
    """Function that matches histogram of a given image with all images in a given directory"""
    img = np.array(img)
    random_images = np.random.choice(os.listdir(ref_dir), size=5)
    for ref in random_images:
        ref_path = ref_dir + '/' + ref
        ref_img = cv.imread(ref_path, cv.IMREAD_ANYDEPTH + cv.IMREAD_GRAYSCALE)
        ref_img = ref_img.reshape(ref_img.shape[0],ref_img.shape[1],1)
        img = match_histograms(img, ref_img)
    #img = torch.from_numpy(img)
    #img = Image.fromarray(img)
    return img

def transform_to_PIL(img):
    transform = transforms.ToPILImage()
    return transform(img)
