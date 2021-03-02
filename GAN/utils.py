import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import ot


from torchvision.models.inception import inception_v3

from scipy.stats import entropy



#%% Functions

def entropic_OT(a, b, M, reg=0.1, maxiter=20, cuda=True):
    """
    Function which computes the autodiff sharp entropic OT loss.
    
    parameters:
        - a : input source measure (TorchTensor (ns))
        - b : input target measure (TorchTensor (nt))
        - M : ground cost between measure support (TorchTensor (ns, nt))
        - reg : entropic ragularization parameter (float)
        - maxiter : number of loop (int)
    
    returns:
        - sharp entropic unbalanced OT loss (float)
    """
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    K = torch.exp(-M/reg).type(Tensor).double()
    v = torch.from_numpy(ot.unif(K.size()[1])).type(Tensor).double()
    
    for i in range(maxiter):
        Kv = torch.matmul(K,v)
        u = a/Kv
        Ku = torch.matmul(torch.transpose(K, 0, 1), u)
        v = b/Ku
        
    pi = torch.matmul(torch.diagflat(u), torch.matmul(K, torch.diagflat(v)))
    return torch.sum(pi*M.double())


def sinkhorn_divergence(X, Y, reg=1000, maxiter=100, cuda=True):
    """
    Function which computes the autodiff sharp Sinkhorn Divergence.
    
    parameters:
        - X : Source data (TorchTensor (batch size, ns))
        - Y : Target data (TorchTensor (batch size, nt))
        - reg : entropic ragularization parameter (float)
        - maxiter : number of loop (int)
    
    returns:
        - sharp Sinkhorn Divergence (float)
    """
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    a = torch.from_numpy(ot.unif(X.size()[0])).type(Tensor).double()
    b = torch.from_numpy(ot.unif(Y.size()[0])).type(Tensor).double()
    M = distances(X, Y) 
    Ms = distances(X, X)
    Mt = distances(Y, Y) 
        
    SD = entropic_OT(a, b, M, reg=reg, maxiter=maxiter)
    SD -= 1./2 * entropic_OT(a, a, Ms, reg=reg, maxiter=maxiter-50)
    SD -= 1./2 * entropic_OT(b, b, Mt, reg=reg, maxiter=maxiter-50)
    return SD


def entropic_OT_loss(X, Y, reg=1000, maxiter=100, cuda=True):
    """
    Function which computes the autodiff sharp Sinkhorn Divergence.
    
    parameters:
        - X : Source data (TorchTensor (batch size, ns))
        - Y : Target data (TorchTensor (batch size, nt))
        - reg : entropic ragularization parameter (float)
        - maxiter : number of loop (int)
    
    returns:
        - Entropic OT loss (float)
    """
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    a = torch.from_numpy(ot.unif(X.size()[0])).type(Tensor).double()
    b = torch.from_numpy(ot.unif(Y.size()[0])).type(Tensor).double()
    M = distances(X, Y)  
        
    OT_loss = entropic_OT(a, b, M, reg=reg, maxiter=maxiter)
    return OT_loss


def emd(X, Y, cuda=True):
    """
    Function which returns optimal trasnportation plan.
    
    parameters:
        - X : Source data (TorchTensor (batch size, ns))
        - Y : Target data (TorchTensor (batch size, nt))
        - reg : entropic ragularization parameter (float)
        - maxiter : number of loop (int)

    returns:
    - Optimal transportation cost (float)
    """
    
    TensorD = torch.cuda.DoubleTensor if cuda else torch.DoubleTensor
    
    a, b = ot.unif(X.size()[0]), ot.unif(Y.size()[0])
    M = distances(X, Y)
    pi = torch.as_tensor(ot.emd(a, b, M.detach().cpu().numpy().copy())).type(TensorD)
    return torch.sum(pi * M.double())



def gen_noise(batch_size, latent_dim, cuda=True):
    """
    Function which returns latent tensor for generator
    
    parameters:
        - batch_size : batch size (int)
        - latent_dim : latent dimension (int)
        - cuda : gpu acceleration (bool)

    returns:
        - N(0,1) distributed tensor (TorchTensor (batch_size, latent_dim))
    """
    
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    return torch.FloatTensor(batch_size, latent_dim, 1, 1).type(Tensor).normal_(0, 1)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """ from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)
    
    
class Sqrt0(torch.autograd.Function):
    """
    Compute the square root and gradients class of a given tensor. Taken from the geomloss package.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Compute the square root for a given Tensor.
        """
        result = input.sqrt()
        result[input < 0] = 0
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the square root gradients of a given Tensor.
        """
        result, = ctx.saved_tensors
        grad_input = grad_output / (2*result)
        grad_input[result == 0] = 0
        return grad_input

def sqrt_0(x):
    """
    Compute the square root for a given Tensor.
    
    parameters:
        - x : Source data (TorchTensor (batch size, ns))

    returns:
        - square roof of x (TorchTensor (batch size, ns))
    """
    return Sqrt0.apply(x)


def squared_distances(x, y):
    """
    Returns the matrix of $\|x_i - y_j\|_2^2$. Taken from the geomloss package.
    
    parameters:
        - x : Source data (TorchTensor (batch size, ns))
        - y : Target data (TorchTensor (batch size, nt))

    returns:
        - Ground cost (float)
    """
    if x.dim() == 2:
        D_xx = (x*x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul( x, y.permute(1,0) )  # (N,D) @ (D,M) = (N,M)
        D_yy = (y*y).sum(-1).unsqueeze(0)  # (1,M)
    elif x.dim() == 3:  # Batch computation
        D_xx = (x*x).sum(-1).unsqueeze(2)  # (B,N,1)
        D_xy = torch.matmul( x, y.permute(0,2,1) )  # (B,N,D) @ (B,D,M) = (B,N,M)
        D_yy = (y*y).sum(-1).unsqueeze(1)  # (B,1,M)
    else:
        print("x.shape : ", x.shape)
        raise ValueError("Incorrect number of dimensions")

    return D_xx - 2*D_xy + D_yy


def distances(x, y):
    """
    Returns the matrix of $\|x_i - y_j\|_2$. Taken from the geomloss package.
    
    parameters:
        - x : Source data (TorchTensor (batch size, ns))
        - y : Target data (TorchTensor (batch size, nt))

    returns:
        - Cost matrix (float)
    """
    return sqrt_0( squared_distances(x,y) )