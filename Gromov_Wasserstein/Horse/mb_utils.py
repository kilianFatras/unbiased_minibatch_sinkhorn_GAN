import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from itertools import combinations, chain
from scipy.special import comb


##############################################################
# Update transportation matrix
##############################################################

def mini_batch(data, weights, batch_size):
    """
     Select a subset of sample uniformly at random without replacement
        parameters : 
        --------------------------------------------------------------
        data : np.array(n, d)
               data
        weights : np.array(n)
                  measure
        batch_size : int
                     minibatch size
    """
    #id = np.random.choice(np.shape(data)[0], batch_size, replace=False, p=weights)
    id = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = weights[id]/np.sum(weights[id]) #ot.unif(batch_size)
    return data[id], sub_weights, id


def update_gamma(gamma, gamma_minibatch, id_a, id_b):
    '''Update mini batch transportation matrix'''
    for i,i2 in enumerate(id_a):
        for j,j2 in enumerate(id_b):
            gamma[i2,j2] += gamma_minibatch[i][j]
    return gamma

