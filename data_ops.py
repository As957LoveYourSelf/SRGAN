import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import glob
import os
import tqdm
import cv2
from torch.utils.data import DataLoader, Dataset

HR_image_path = "/dataset/HR"
save_LR_image_path = "/dataset/LR"
train_image_path = "/dataset/train"
test_image_path = "/dataset/test"

if not os.path.exists(save_LR_image_path):
    os.mkdir(save_LR_image_path)
    os.mkdir(train_image_path)
    os.mkdir(test_image_path)


def hr2lr(HR_path):
    hr_images = glob.glob(HR_path)
    for hr_image in hr_images:

        image = cv2.imread(hr_image)
        image = transforms.ToTensor()(image)



