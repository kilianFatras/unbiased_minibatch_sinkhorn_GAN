import numpy as np
import matplotlib.pylab as pl
import ot
from utils import mini_batch, update_gamma



def get_div_mbot_without_r(a, b, all_comb_s, all_comb_t, bs_s, bs_t, M, M_s, M_t, p):
    '''Compute the divergence MBOT without replacement between 2 measures
    
       Parameters:
       -------------------------------------------
        a : ndarray, shape (ns,)
            Source measure.
        b : ndarray, shape (nt,)
            Target measure.
        all_comb_s : ndarray, shape (ns**m,m)
                    Source minibatch without replacement
        all_comb_t : ndarray, shape (nt**m,m)
                    Target minibatch without replacement
        bs_s : int
               Source minibatch size
        bs_t : int
               Target minibatch size
        M : ndarray, shape (ns, nt)
            Cost matrix between source and target domains.
        M_s : ndarray, shape (ns, ns)
              Cost matrix between source and source domains.
        M_t : ndarray, shape (nt, nt)
              Cost matrix between target and target domains.
        p : int
            power of Wasserstein distance

       Returns
       --------------------------------------
        value of MBOT divergence without replacement 
    '''

    OT_ab = get_mbot_without_r(a, b, all_comb_s, all_comb_t, bs_s, bs_t, M, p)
    OT_s = get_mbot_without_r(a, a, all_comb_s, all_comb_s, bs_s, bs_s, M_s, p)
    OT_t = get_mbot_without_r(b, b, all_comb_t, all_comb_t, bs_t, bs_t, M_t, p)
    return OT_ab - 1./2 * (OT_s + OT_t)


def get_mbot_without_r(a, b, all_comb_s, all_comb_t, bs_s, bs_t, M, p):
    '''Compute the expectation of MBOT without replacement

       Parameters:
       -------------------------------------------
        a : ndarray, shape (ns,)
            Source measure.
        b : ndarray, shape (nt,)
            Target measure.
        all_comb_s : ndarray, shape (ns**m,m)
                    Source minibatch without replacement
        all_comb_t : ndarray, shape (nt**m,m)
                    Target minibatch without replacement
        bs_s : int
               Source minibatch size
        bs_t : int
               Target minibatch size
        M : ndarray, shape (ns, nt)
            Cost matrix.
        p : int
            power of Wasserstein distance

       Returns
       --------------------------------------
        value of MBOT without replacement 
    '''

    nb_comb_s = np.shape(all_comb_s)[0]
    nb_comb_t = np.shape(all_comb_t)[0]
    cost = 0
    norm_coeff = 0
    for i in range(nb_comb_s):
        id_a = all_comb_s[i]
        for j in range(nb_comb_t):
            id_b = all_comb_t[j]
            sub_weights_a = a[id_a]/np.sum(a[id_a])
            sub_weights_b = b[id_b]/np.sum(b[id_b])

            sub_M = M[id_a, :][:, id_b].copy()
            G0 = np.power(ot.emd2(sub_weights_a, sub_weights_b, sub_M), 1./p) #gives Wasserstein_p0^p where p0 is the power of cost matrix

            #Test update gamma
            full_weight = np.sum(a[id_a]) * np.sum(b[id_b])
            cost += full_weight * G0
            norm_coeff += full_weight

    return (1/norm_coeff) * cost
