import torch.nn as nn



class Feature_extractor(nn.Module):
    """
    Feature extractor class. Taken from https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    """
    def __init__(self, img_size, n_channels, k=100, ndf=64):
        super(Feature_extractor, self).__init__()
        assert img_size % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial_conv_{0}-{1}'.format(n_channels, ndf),
                        nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = img_size / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
    
    
class Generator(nn.Module):
    """
    Generator class. Taken from https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    """
    def __init__(self, img_size, n_channels, k=100, ngf=64):
        super(Generator, self).__init__()
        assert img_size % 16 == 0, "img_size has to be a multiple of 16"

        cngf, timg_size = ngf // 2, 4
        while timg_size != img_size:
            cngf = cngf * 2
            timg_size = timg_size * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(k, cngf), nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < img_size // 2:
            main.add_module('pyrami_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final_{0}-{1}_convt'.format(cngf, n_channels), nn.ConvTranspose2d(cngf, n_channels, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(n_channels), nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
    

def weights_init(m):
    """
    Init neural network layers. Taken from https://github.com/OctoberChang/MMD-GAN/blob/master/base_module.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)