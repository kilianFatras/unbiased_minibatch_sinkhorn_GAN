import numpy as np
import ot

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from architecture import Feature_extractor, Generator, weights_init
from utils import squared_distances, emd, gen_noise, sinkhorn_divergence, inception_score, IgnoreLabelDataset

img_size = 32
channels = 3
latent_dim = 100

img_shape = (channels, img_size, img_size)
PATH = 'results/umbSD_euclidean_reg_100.0_epoch975.pth'
print('loaded method : ', PATH)
netG = Generator(img_size, channels).cuda()
checkpoint = torch.load(PATH)

netG.load_state_dict(checkpoint['generator_state_dict'])

netG.eval()
z = Variable(gen_noise(20000, latent_dim, cuda=True))
gen_imgs = netG(z)
print ("Calculating Inception Score...")
print (inception_score(gen_imgs, cuda=True, batch_size=32, resize=True, splits=10))
