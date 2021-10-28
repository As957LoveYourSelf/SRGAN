import torchvision
import torch
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data.dataset import T_co
from torchvision.utils import save_image
import torch.nn.functional as F
import glob
import os
import tqdm
import cv2
from torch.utils.data import Dataset

HR_image_path = "/dataset/HR"
train_lr_image_path = "/dataset/train/LR"
train_hr_image_path = "/dataset/train/HR"
test_lr_image_path = "/dataset/test/LR"
test_hr_image_path = "/dataset/test/HR"

if not os.path.exists(train_hr_image_path):
    os.mkdir(test_hr_image_path)
    os.mkdir(test_lr_image_path)
    os.mkdir(train_lr_image_path)
    os.mkdir(train_hr_image_path)

tran = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(96),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def hr2lr(tensor: Tensor, scale_factor):
    assert (isinstance(tensor, Tensor))
    return F.interpolate(tensor, scale_factor=scale_factor, mode="bilinear")


def save_hr2lr(hr_image, save_hr_path, save_lr_path):
    filename = os.path.basename(hr_image)
    image = cv2.imread(hr_image)
    image = tran(image)
    # save HR image
    save_image(image, os.path.join(save_hr_path, "hr_" + filename))
    image = hr2lr(image, scale_factor=0.25)
    # save LR image
    save_image(image, os.path.join(save_lr_path, "lr_" + filename))


def make_dateset(hr_image_path):
    hr_images = glob.glob(hr_image_path + "/*")
    l = len(hr_images)
    train_len = l*0.8
    print(f"Begin Build Train DataSet, Length = {train_len}")
    for hr_image in tqdm.tqdm(hr_images[:train_len]):
        save_hr2lr(hr_image, train_hr_image_path, train_lr_image_path)
    print(f"Begin Build Test DataSet, Length = {l-train_len}")
    for hr_image in tqdm.tqdm(hr_images[train_len:]):
        save_hr2lr(hr_image, test_hr_image_path, test_lr_image_path)


class PreDateSet(Dataset):
    def __init__(self):
        super(PreDateSet, self).__init__()

    def __getitem__(self, index) -> T_co:
        pass

    def __len__(self):
        pass


def main():
    pass



if __name__ == '__main__':
    main()