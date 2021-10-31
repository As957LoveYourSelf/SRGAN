import os
import sys

import tqdm
from torchvision.utils import save_image
from net import Generator, Discriminator
import torch
from torchvision.models import vgg19
from torch.utils.data import DataLoader
from data_ops import PreDateSet
import loss_functions as losses
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter

# logger = SummaryWriter(log_dir="result/log")

vggnet = vgg19()
pretrainmodel_path = "pretrainmodels/vgg19.pth"
pretrainmodel = torch.load(pretrainmodel_path)
vggnet.load_state_dict(pretrainmodel)
vggnet = vggnet.features
vggnet.cuda()
for p in vggnet.parameters():
    p.requires_grad = False

# load model
g_net = Generator(residual_num=19, upsample_x=2)
g_net.cuda()
d_net = Discriminator()
d_net.cuda()

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
lr_g = 1e-5
lr_d = 1e-5

optimizer_G = Adam(g_net.parameters(), lr=lr_g)
optimizer_D = Adam(d_net.parameters(), lr=lr_d)


train_predataset = PreDateSet(train_hr_image_path, train_lr_image_path)
train_data = DataLoader(train_predataset, batch_size=BATCH_SIZE, shuffle=True)
test_predataset = PreDateSet(test_hr_image_path, test_lr_image_path)
test_data = DataLoader(test_predataset, batch_size=BATCH_SIZE, shuffle=False)
test_data = iter(test_data)
train_data_l = len(train_data)

for epoch in range(1, EPOCH+1):
    for i, (hr_image, lr_image) in tqdm.tqdm(enumerate(train_data, start=1)):

        d_net.train()
        g_net.train()
        ###################
        # Train Generator #
        ###################
        # gen_loss_1 = mse_loss + adv_loss
        # gen_loss_2 = vgg_loss + adv_loss
        optimizer_G.zero_grad()
        g_hr_img = g_net(lr_image)
        disc_g_img = d_net(g_hr_img)
        # 3 parts losses
        content_loss = losses.mse_loss(hr_image, g_hr_img).mean()
        vgg_loss = losses.vgg_loss(vggnet(hr_image), vggnet(g_hr_img)).mean()
        adv_loss = losses.adv_loss(disc_g_img).mean()

        gen_loss = content_loss + 1e-3*vgg_loss + 1e-6*adv_loss
        gen_loss.backward()
        optimizer_G.step()

        #######################
        # Train Discriminator #
        #######################
        optimizer_D.zero_grad()

        real_d_img = d_net(hr_image)
        disc_loss = 1. - disc_g_img.mean().detach() + real_d_img.mean()
        disc_loss.backward()
        optimizer_D.step()

        # print train status
        print(f"[{epoch}/{EPOCH}Epoch][{i}/{train_data_l}Iter]=>G_loss: {gen_loss.item()}|D_predict: "
              f"{disc_loss.item()}")

        # save result (images)
        if i % 2000 == 0:
            g_net.eval()
            d_net.eval()
            hr_img, lr_img = next(test_data)
            with torch.no_grad():
                fake_hr_img = g_net(lr_img)
                predict_img = d_net(fake_hr_img)
            res = torch.cat([fake_hr_img, hr_img], dim=0)
            res = res.cpu()
            print("Begin save image...")
            save_image(res, os.path.join(save_image_path, f"{epoch}_epoch_{i}_iter.jpg"))
            print("save success.")
    gen_model_name = f"{epoch}_epoch_gen_model.pth"
    disc_model_name = f"{epoch}_epoch_disc_model.pth"
    print("Saving generate model...")
    torch.save(g_net.state_dict(), os.path.join(save_model_path, gen_model_name))
    print("Saving discriminate model...")
    torch.save(d_net.state_dict(), os.path.join(save_model_path, disc_model_name))
