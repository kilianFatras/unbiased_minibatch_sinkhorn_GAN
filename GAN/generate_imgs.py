import argparse
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
import matplotlib.pyplot as plt

def imshow(ax, img):
#    img = img.view(64, 3, 32, 32) #check shape in torch example
    img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_fig(ax, x, method):
    imshow(ax, x)
    ax.set_title(method, fontsize=22)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    plt.tight_layout()
    
    
img_size = 32
channels = 3
latent_dim = 100

img_shape = (channels, img_size, img_size)
z = Variable(gen_noise(64, latent_dim, cuda=True))

#PATH = 'umbSD_weights_1000_euc_64.pth'
PATH = 'SD_weights_10.pth'
#PATH = 'umbW_weights_euc.pth'
PATH = 'wgan_gp2.pth'
#methods = ['SD_weights_10.pth', 'SD_weights_100.pth', 'SD_weights_1000.pth', 'umbSD_weights_10_euc_64.pth', 'umbSD_weights_100_euc_64.pth', 'umbSD_weights_1000_euc_64.pth', 'umbW_weights_euc.pth', 'wgan_gp2.pth']
methods = ['SD_weights_100.pth', 'umbSD_weights_100_euc_64.pth', 'wgan_gp2.pth']
name = [r"SD $\varepsilon$=100", r"$\Lambda_{SD}, \varepsilon=100$", 'WGAN-GP']

fig = plt.figure(figsize=(20,8))

for i in range(len(methods)):
    ax = plt.subplot(1, 3, i+1)
    PATH = methods[i]
    print('loaded method : ', PATH)
    netG = Generator(img_size, channels).cuda()
    checkpoint = torch.load(PATH)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netG.eval()
    gen_imgs = netG(z)
    
    plot_fig(ax, torchvision.utils.make_grid(gen_imgs), name[i])

plt.tight_layout()
plt.savefig('imgs/gan_paper.pdf')
    
    #save_image(gen_imgs.data, "imgs/{}.png".format(PATH), nrow=8, normalize=True)