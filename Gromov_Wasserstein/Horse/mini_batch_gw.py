import numpy as np
import ot
from mb_utils import mini_batch, update_gamma


def calculate_stoc_gamma(xs, xt, a, b, bs_s, bs_t, num_iter, M1, M2,
                        lambd=1e-1, method='emd'):
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
        M1 : ndarray, shape (ns, nt)
             Source cost matrix.
        M2 : ndarray, shape (ns, nt)
             Target cost matrix.
        lambd : float
                entropic reg value
        method : str
                 name of the method

       Returns
       --------------------------------------
       Incomplete MBOT plan with replacement 
    '''
    stoc_gamma = np.zeros((np.shape(xs)[0], np.shape(xt)[0]))
    error_norm = []
    error_norm2 = []
    sum_weights = 0
    for i in range(num_iter):
        #Test mini batch
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, bs_s)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, bs_t)

        if method == 'emd':
            sub_M1 = M1[id_a, :][:, id_a].copy()
            sub_M2 = M2[id_b, :][:, id_b].copy()
            G0 = ot.gromov.gromov_wasserstein(sub_M1, sub_M2,
                sub_weights_a, sub_weights_b, 'square_loss')

        elif method == 'entropic':
            sub_M1 = M1[id_a, :][:, id_a]
            sub_M2 = M2[id_b, :][:, id_b]
            G0 = ot.gromov.entropic_gromov_wasserstein(sub_M1, sub_M2,
            sub_weights_a, sub_weights_b, 'square_loss', epsilon=lambd)

        #Test update gamma
        cur_weight = np.sum(a[id_a]) * np.sum(b[id_b])
        stoc_gamma = update_gamma(stoc_gamma, cur_weight * G0, id_a, id_b)
        sum_weights += cur_weight

    return (1/sum_weights) * stoc_gamma


def calculate_stoc_gamma2(xs, xt, a, b, bs_s, bs_t, num_iter, M1, M2,
                        lambd=1e-1, method='emd'):
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
        M1 : ndarray, shape (ns, nt)
             Source cost matrix.
        M2 : ndarray, shape (ns, nt)
             Target cost matrix.
        lambd : float
                entropic reg value
        method : str
                 name of the method

       Returns
       --------------------------------------
       Incomplete MBOT with replacement value
    '''

    avg_gw_val = 0
    sum_weights = 0
    for i in range(num_iter):
        #Test mini batch
        sub_xs, sub_weights_a, id_a = mini_batch(xs, a, bs_s)
        sub_xt, sub_weights_b, id_b = mini_batch(xt, b, bs_t)

        if method == 'emd':
            sub_M1 = M1[id_a, :][:, id_a].copy()
            sub_M2 = M2[id_b, :][:, id_b].copy()
            val_GW = ot.gromov.gromov_wasserstein2(sub_M1, sub_M2,
                sub_weights_a, sub_weights_b, 'square_loss')

        elif method == 'entropic':
            sub_M1 = M1[id_a, :][:, id_a]
            sub_M2 = M2[id_b, :][:, id_b]
            val_GW = ot.gromov.entropic_gromov_wasserstein2(sub_M1, sub_M2,
            sub_weights_a, sub_weights_b, 'square_loss', epsilon=lambd)

        #Test update gamma
        cur_weight = np.sum(a[id_a]) * np.sum(b[id_b])
        avg_gw_val += cur_weight * val_GW
        sum_weights += cur_weight

    return (1/sum_weights) * avg_gw_val
