from net import Generator, Discriminator
import torch
from torchvision.models import vgg19
import torchsummary as summary

vggnet = vgg19()
pretrainmodel_path = "pretrainmodels/vgg19.pth"
pretrainmodel = torch.load(pretrainmodel_path)
vggnet.load_state_dict(pretrainmodel)

# load model
g_net = Generator(residual_num=19, upsample_x=2)
d_net = Discriminator()
print("G_net model summary:")
summary.summary(g_net, input_size=(3, 24, 24))
print("D_net model summary:")
summary.summary(d_net, input_size=(3, 96, 96))
# define parameters
EPOCH = 10000









