from net import Generator, Discriminator
import torch
import torchsummary as summary

device = "cuda" if torch.cuda.is_available() else "cpu"

g_net = Generator(residual_num=19, upsample_x=2)
g_net.to(device)
d_net = Discriminator()
d_net.to(device)
print("G_net model summary:")
summary.summary(g_net, input_size=(3, 24, 24))
print("D_net model summary:")
summary.summary(d_net, input_size=(3, 96, 96))
