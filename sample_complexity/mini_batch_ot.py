import numpy as np
import matplotlib.pylab as pl
import ot
from utils import mini_batch
import scipy



def calculate_inc_mbot(xs, xt, a, b, bs_s, bs_t, num_iter, M):
    '''
        Compute the incomplete MBOT without replacement
    
       Parameters:
       -------------------------------------------
        xs : ndarray, shape (ns, d)
            Source data.
        xt : ndarray, shape (nt, d)
            Target data.
        a : ndarray, shape (ns,)
            Source measure.
        b : ndarray, shape (nt,)
            Target measure.
        bs_s : int
               Source minibatch size
        bs_t : int
               Target minibatch size
        num_iter : int
            number of iterations
        M : ndarray, shape (ns, nt)
            Cost matrix.

       Returns
       --------------------------------------
        value of MBOT with replacement 
    '''
    cost = 0
    norm_coeff = 0
    for i in range(num_iter):
        #Test mini batch
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, bs_s)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, bs_t)

        sub_M = M[id_a,:][:,id_b].copy()
        cur_mbot = ot.emd2(sub_weights_a, sub_weights_b, sub_M)

        #Test update gamma
        full_weight = np.sum(a[id_a]) * np.sum(b[id_b])
        cost += full_weight * cur_mbot
        norm_coeff += full_weight

    return (1/norm_coeff) * cost



