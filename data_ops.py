import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import glob
import os
import tqdm
import cv2
from torch.utils.data import Dataset

HR_image_path = "dataset/HR"
train_lr_image_path = "dataset/train/LR"
train_hr_image_path = "dataset/train/HR"
test_lr_image_path = "dataset/test/LR"
test_hr_image_path = "dataset/test/HR"

if not os.path.exists(train_hr_image_path):
    os.makedirs(test_hr_image_path)
    os.makedirs(test_lr_image_path)
    os.makedirs(train_lr_image_path)
    os.makedirs(train_hr_image_path)

tran = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(96)
    ]
)


def hr2lr(image, scale_factor):
    t_shape = image.shape
    image = image.view((1, *list(t_shape))).cuda()
    return F.interpolate(image, scale_factor=scale_factor)


def save_hr2lr(hr_image, save_hr_path, save_lr_path):
    filename = os.path.basename(hr_image)
    image = cv2.imread(hr_image)
    image = tran(image).cuda()
    # save HR image
    save_image(image, os.path.join(save_hr_path, "hr_" + filename))
    image = hr2lr(image, scale_factor=0.25)
    # save LR image
    save_image(image, os.path.join(save_lr_path, "lr_" + filename))


def make_dateset(hr_image_path):
    hr_images = glob.glob(hr_image_path+"/*")
    l = len(hr_images)
    train_len = int(l * 0.8)
    print(f"Begin Build Train DataSet, Length = {train_len}")
    for hr_image in tqdm.tqdm(hr_images[:train_len]):
        try:
            save_hr2lr(hr_image, train_hr_image_path, train_lr_image_path)
        except ValueError:
            continue
    print(f"Begin Build Test DataSet, Length = {l - train_len}")
    for hr_image in tqdm.tqdm(hr_images[train_len:]):
        try:
            save_hr2lr(hr_image, test_hr_image_path, test_lr_image_path)
        except ValueError:
            continue


class PreDateSet(Dataset):
    def __init__(self, HR_path, LR_path):
        super(PreDateSet, self).__init__()
        hr_images = glob.glob(os.path.join(HR_path, "*"))
        lr_images = glob.glob(os.path.join(LR_path, "*"))
        self.images_pair_data = list(zip(hr_images, lr_images))

    def __getitem__(self, index):
        tran_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        hr_img, lr_img = self.images_pair_data[index]
        hr_img, lr_img = tran_(cv2.imread(hr_img)).cuda(), tran_(cv2.imread(lr_img)).cuda()
        return hr_img, lr_img

    def __len__(self):
        return len(self.images_pair_data)

