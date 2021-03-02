import argparse
import os
import numpy as np
import ot

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.data import DataLoader
import random 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from architecture import Feature_extractor, Generator, weights_init
from utils import squared_distances, emd, gen_noise, sinkhorn_divergence

def train(config):
    
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    latent_dim = config["latent_dim"]
    img_size = config["img_size"]
    channels = config["channels"]
    clip_value = config["clip_value"]
    sample_interval = config["sample_interval"]
    n_critic = config["n_critic"]
    k = config["k"]
    reg = config["reg"]
    print("n_epochs, batch_size, lr, latent_dim, img_size, channels, clip_value, sample_interval, n_critic, k, reg :\n", n_epochs, batch_size, lr, latent_dim, img_size, channels, clip_value, sample_interval, n_critic, k, reg)

    img_shape = (channels, img_size, img_size)

    cuda = True if torch.cuda.is_available() else False
    print("CUDA is avalaible :", cuda)
    
    
    # Initialize generator
    netG = Generator(img_size, channels)
    netD = Feature_extractor(img_size, channels)
    if cuda:
        netG.cuda()
        netD.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorD = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor

    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    netG.apply(weights_init)
    netD.apply(weights_init)

    # Configure data loader
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Optimizers
    optimizer_D = torch.optim.RMSprop(netD.parameters(), lr=lr)
    optimizer_G = torch.optim.RMSprop(netG.parameters(), lr=lr)


    # ----------
    #  Training
    # ----------

    batches_done = 0
    for epoch in range(n_epochs):

        data_iter = iter(dataloader)
        i = 0
        while (i < len(dataloader)):

            for j in range(n_critic):
                if i == len(dataloader):
                    break

                (imgs, _) = data_iter.next()
                if i == 0:
                    prec_imgs = imgs
                i += 1

                ########################
                # TRAIN Cost 
                ########################
                optimizer_D.zero_grad()
                # Configure input

                prec_real_imgs = Variable(prec_imgs.type(Tensor))
                real_imgs = Variable(imgs.type(Tensor))
                loss_G_item = 0
                netD.zero_grad()

                # ----- True Images -----
                batch_size = real_imgs.size(0)
                old_batch_size = prec_real_imgs.size(0)
                features_real_imgs = netD(real_imgs).view(batch_size, -1)
                features_prec_real_imgs = netD(prec_real_imgs).view(old_batch_size, -1)

                # ----- Generated Images -----
                noise = torch.FloatTensor(batch_size, latent_dim, 1, 1).type(Tensor).normal_(0, 1)
                G_imgs = Variable(netG(noise), requires_grad=False)  # Freeze G_imgs gradient

                noise2 = torch.FloatTensor(batch_size, latent_dim, 1, 1).type(Tensor).normal_(0, 1)
                G_imgs2 = Variable(netG(noise2), requires_grad=False)  # Freeze G_imgs gradient

                features_G_imgs = netD(G_imgs).view(batch_size, -1)
                features_G_imgs2 = netD(G_imgs2).view(batch_size, -1)

                # ----- Loss -----
                loss_D_ab = sinkhorn_divergence(features_real_imgs, features_G_imgs, reg=reg, cuda=cuda)  # U(a, b)
                loss_D_aa = sinkhorn_divergence(features_prec_real_imgs, features_real_imgs, reg=reg, cuda=cuda)  # U(a,a)
                loss_D_bb = sinkhorn_divergence(features_G_imgs, features_G_imgs2, reg=reg, cuda=cuda)  # U(b,b)

                loss_D = loss_D_ab - 1./2 * loss_D_aa - 1./2 * loss_D_bb


                loss_D.backward(mone)  # mone -> loss_D * -1
                optimizer_D.step()

                for p in netD.parameters(): #Clamp decoder
                    p.data.clamp_(-0.01, 0.01)

                prec_imgs = imgs    

            ########################
            # TRAIN GENERATOR
            ########################
            optimizer_G.zero_grad()
            for _ in range(k):

                # Sample noise as generator input
                z = Variable(gen_noise(batch_size, latent_dim, cuda=cuda), requires_grad=True)
                gen_imgs = netG(z)
                feature_gen_imgs = netD(gen_imgs).view(batch_size, -1)

                z2 = Variable(gen_noise(batch_size, latent_dim, cuda=cuda), requires_grad=True)
                feature_gen_imgs2 = netD(netG(z2)).view(batch_size, -1)

                loss_ab = sinkhorn_divergence(feature_gen_imgs, features_real_imgs.detach(), reg=reg, cuda=cuda)  # U(a,b)
                loss_aa = sinkhorn_divergence(feature_gen_imgs, feature_gen_imgs2, reg=reg, cuda=cuda)  # U(a,a)

                loss_G = (loss_ab - 1./2 * loss_aa)/k
                loss_G_item += loss_G.item()
                loss_G.backward(one)

            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, batches_done % (len(dataloader)/n_critic), (len(dataloader)/n_critic), loss_D.item(), loss_G.item())
            )

            batches_done += 1
        if epoch%sample_interval==0:
            save_image(gen_imgs.data[:64], 
                       os.path.join(config["output_path_imgs"], "epoch_{}.png".format(epoch)), 
                       nrow=8, normalize=True)
            
    torch.save({
                'generator_state_dict': netG.state_dict(),
                'features_state_dict': netD.state_dict(),
                'optimizer_gen_state_dict': optimizer_G.state_dict(),
                'features_state_dict': optimizer_D.state_dict(),
                }, os.path.join(config["output_path_results"], 
                                'umbSD_euclidean_reg_{}_final.pth'.format(reg)))

    

if __name__ == "__main__":
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Unbiased MiniBatch Sinkhorn Divergence GAN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--sample_interval', type=int, default=25, help="interval of two continuous output model")
    parser.add_argument('--output_dir_imgs', type=str, default='imgs/', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--output_dir_results', type=str, default='results/', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--n_epochs', type=int, default=1000, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.00005, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="training batch size")
    parser.add_argument('--latent_dim', type=int, default=100, help="noise's latent dim")
    parser.add_argument('--img_size', type=int, default=32, help="image size")
    parser.add_argument('--channels', type=int, default=3, help="image channel")
    parser.add_argument('--clip_value', type=float, default=0.01, help="training batch size")
    parser.add_argument('--n_critic', type=int, default=5, help="number of critic update")
    parser.add_argument('--k', type=int, default=1, help="number of minibatch couple")
    parser.add_argument('--reg', type=float, default=100, help="entropic regularization constant")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    torch.backends.cudnn.benchmark=True

    # train config
    config = {}
    config['args'] = args
    config["gpu"] = args.gpu_id
    config["n_epochs"] = args.n_epochs
    config["lr"] = args.lr
    config["batch_size"] = args.batch_size
    config["latent_dim"] = args.latent_dim
    config["img_size"] = args.img_size
    config["channels"] = args.channels
    config["clip_value"] = args.clip_value
    config["n_critic"] = args.n_critic
    config["k"] = args.k
    config["reg"] = args.reg
    config["sample_interval"] = args.sample_interval
    config["output_path_imgs"] = args.output_dir_imgs
    config["output_path_results"] = args.output_dir_results
    
    
    if os.path.exists(config["output_path_imgs"]):
        print("imgs dir exists")
    else:
        print("creating imgs dir")
        os.mkdir(config["output_path_imgs"])
    if os.path.exists(config["output_path_results"]):
        print("results dir exists")
    else:
        print("creating results dir")
        os.mkdir(config["output_path_results"])


    train(config)