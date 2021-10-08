import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from itertools import combinations, chain
from scipy.special import comb



##############################################################
# SD
##############################################################
def empirical_sinkhorn_loss(a, b, X, Y, reg, numIterMax = 10000):
    '''
    Compute the sinkhorn_loss distance
    '''

    M = ot.dist(X, Y)
    pi = ot.sinkhorn(a, b, M, reg, numItermax=numIterMax)
    return np.sum(M*pi)

def empirical_sinkhorn_divergence(a, b, X, Y, reg):
    '''
    Compute the sinkhorn divergence
    '''

    sinkhorn_div = (empirical_sinkhorn_loss(a, b, X, Y, reg)
                    - 1/2 * empirical_sinkhorn_loss(a, a, X, X, reg)
                    - 1/2 * empirical_sinkhorn_loss(b, b, Y, Y, reg))
    return max(0, sinkhorn_div)

##############################################################
# Update transportation matrix
##############################################################

def mini_batch(data, weights, batch_size):
    """select a subset of sample space according to measure
        arg : data
    """
    #id = np.random.choice(np.shape(data)[0], batch_size, replace=False, p=weights)
    id = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = weights[id]/np.sum(weights[id]) #ot.unif(batch_size)
    return data[id], sub_weights, id

#@numba.jit(parallel = True)
def update_gamma(gamma, gamma_minibatch, id_a, id_b):
    '''Update mini batch transportation matrix'''
    for i,i2 in enumerate(id_a):
        for j,j2 in enumerate(id_b):
            gamma[i2,j2] += gamma_minibatch[i][j]
    return gamma


##############################################################
# Combination functions
##############################################################

def comb_index(n, k):
    """Return all possible combinaisons from nCp"""
    count = comb(n, k, exact=True)
    index = np.fromiter(chain.from_iterable(combinations(range(n), k)), int,
            count=count*k)
    return index.reshape(-1, k)


##############################################################
# Plot functions
##############################################################

def plot_OT_mat(xs, xt, gamma, title):
    '''Plot the transportation matrix'''
    pl.figure()
    print(np.sum(gamma))
    ot.plot.plot2D_samples_mat(xs, xt, gamma, c=[.5, .5, 1])
    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    pl.legend(loc=0)
    pl.title(title)
    pl.savefig('imgs/'+title)
    pl.show()


def imshow_OT(gamma, title):
    pl.imshow(gamma, interpolation='nearest')
    pl.title(title)
    pl.savefig('imgs/'+title)
    pl.show()


def plot_norm_error(tab, max_iter, title, file_name):
    x = np.linspace(0, max_iter, len(tab))
    pl.loglog(x, tab)
    pl.title(title)
    pl.xlabel('iter')
    pl.ylabel('norm error')
    pl.savefig('imgs/'+file_name)
    pl.show()
