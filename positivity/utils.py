import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from itertools import combinations, permutations, chain, product
from scipy.special import comb



##############################################################
# Update transportation matrix
##############################################################

def mini_batch(data, weights, batch_size):
    """select a subset of sample space according to measure
        arg : data
    """
    id = np.random.choice(np.shape(data)[0], batch_size, replace=False, p=weights)
    sub_weights = ot.unif(batch_size)
    return data[id], sub_weights, id


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
    """Return all possible combinaisons without replacement"""
    index = np.array(list(combinations(range(n), k)))
    return index

def all_comb_permutation(n, k):
    """Return all possible permuations of combinaisons without replacement"""
    index = np.array(list(permutations(range(n), k)))
    return index

def all_arr_w_replacement(n, k):
    """Return all possible arrangement with replacement"""
    index = np.array(list(product(range(n), repeat=k)))
    return index


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


def imshow_OT(gamma, title=None):
    pl.imshow(gamma, interpolation='nearest')
    if title is not None:
        pl.title(title)
        pl.savefig('imgs/'+title)
    pl.show()
