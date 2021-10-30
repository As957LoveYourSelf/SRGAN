import os
import numpy as np
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import save_image
from net import Generator, Discriminator
import torch
from torchvision.models import vgg19
import torchsummary as summary
from torch.utils.data import DataLoader
from data_ops import PreDateSet
import loss_functions as losses
from torch.optim import Adam, SGD

device = "cuda" if torch.cuda.is_available() else "cpu"

vggnet = vgg19()
pretrainmodel_path = "pretrainmodels/vgg19.pth"
pretrainmodel = torch.load(pretrainmodel_path)
vggnet.load_state_dict(pretrainmodel)
vggnet = vggnet.features
vggnet.to(device)
for p in vggnet.parameters():
    p.requires_grad = False

# load model
g_net = Generator(residual_num=19, upsample_x=2)
d_net = Discriminator()
print("G_net model summary:")
summary.summary(g_net, input_size=(3, 24, 24))
print("D_net model summary:")
summary.summary(d_net, input_size=(3, 96, 96))

train_lr_image_path = "dataset/train/LR"
train_hr_image_path = "dataset/train/HR"
test_lr_image_path = "dataset/test/LR"
test_hr_image_path = "dataset/test/HR"
save_image_path = "result/images"
save_model_path = "result/models"
save_loss_path = "result/loss"

if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)
    os.makedirs(save_model_path)
    os.makedirs(save_loss_path)

# define parameters
EPOCH = 10000
BATCH_SIZE = 8
lr_g = 1e-4
lr_d = 1e-4

optimizer_G = Adam(g_net.parameters(), lr=lr_g)
optimizer_D = Adam(d_net.parameters(), lr=lr_d)


train_predataset = PreDateSet(train_hr_image_path, train_lr_image_path)
train_data = DataLoader(train_predataset, batch_size=BATCH_SIZE, shuffle=True)
test_predataset = PreDateSet(test_hr_image_path, test_lr_image_path)
test_data = DataLoader(test_predataset, batch_size=BATCH_SIZE, shuffle=False)
test_data = iter(test_data)
train_data_l = len(train_predataset)
test_data_l = len(test_predataset)

for epoch in range(EPOCH):
    for i, (hr_image, lr_image) in enumerate(train_predataset):

        d_net.train()
        g_net.train()
        ###################
        # Train Generator #
        ###################
        # gen_loss_1 = mse_loss + adv_loss
        # gen_loss_2 = vgg_loss + adv_loss
        optimizer_G.zero_grad()
        g_hr_img = g_net(lr_image).to(device)
        disc_g_img = d_net(g_hr_img).to(device)
        # 3 parts losses
        content_loss = losses.mse_loss(hr_image, g_hr_img)
        vgg_loss = losses.vgg_loss(vggnet(hr_image), vggnet(g_hr_img))
        adv_loss = losses.adv_loss(disc_g_img)

        gen_loss = content_loss + 1e-3*vgg_loss + 1e-6*adv_loss

        gen_loss.backward()
        optimizer_G.step()

        #######################
        # Train Discriminator #
        #######################
        optimizer_D.zero_grad()

        real_d_img = d_net(hr_image).to(device)
        disc_loss = 1 - disc_g_img.mean() + real_d_img.mean()
        disc_loss.backward()
        optimizer_D.step()

        # print train status
        print(f"[{epoch}/{EPOCH}Epoch]=>G_loss: {gen_loss.item}|D_predict: {disc_loss.item}|[{i}/{train_data_l}]Iter")

        # save result (images)
        g_net.eval()
        d_net.eval()
        if i % 100 == 0:
            hr_img, lr_img = next(test_data)
            with torch.no_grad:
                fake_hr_img = g_net(lr_img)
                predict_img = d_net(fake_hr_img)
            res = torch.cat([fake_hr_img, hr_img], dim=0)
            res = res.cpu()
            print("Begin save image...")
            save_image(fake_hr_img, save_image_path+f"{epoch}_epoch"+f"_{i}_iter_predict_{d_net}"+".jpg")
            print("save success.")
        gen_model_name = f"{epoch}_epoch_gen_model.pth"
        disc_model_name = f"{epoch}_epoch_disc_model.pth"
        print("Saving generate model...")
        torch.save(g_net.state_dict(), gen_model_name)
        print("Saving discriminate model...")
        torch.save(d_net.state_dict(), disc_model_name)
