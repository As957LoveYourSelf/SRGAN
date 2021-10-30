import torch.nn.functional as F


def mse_loss(hr_feature, generate_hr_feature):
    return F.mse_loss(generate_hr_feature, hr_feature)


def vgg_loss(vgg_hr, vgg_generate_hr):
    return F.mse_loss(vgg_generate_hr, vgg_hr)


def adv_loss(d_g_hr_features):
    return F.sigmoid(d_g_hr_features)
