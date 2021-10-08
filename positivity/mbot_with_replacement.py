import numpy as np
import ot
from utils import update_gamma


def get_mbot_with_r(a, b, all_arr_s, all_arr_t, bs_s, bs_t, M, p):
    '''Compute the expectation of MBOT with replacement

       Parameters:
       -------------------------------------------
        a : ndarray, shape (ns,)
            Source measure.
        b : ndarray, shape (nt,)
            Target measure.
        all_arr_s : ndarray, shape (ns**m,m)
                    Source minibatch with replacement
        all_arr_t : ndarray, shape (nt**m,m)
                    Target minibatch with replacement
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
        value of MBOT with replacement 
    '''
    cost = 0
    for id_a in all_arr_s:
        for id_b in all_arr_t:
            # proba to pick this batch couple
            proba_couple_batches = np.prod(a[id_a]) * np.prod(b[id_b])

            sub_M = M[id_a, :][:, id_b].copy()
            wass = np.power(ot.emd2(ot.unif(bs_s), ot.unif(bs_t), sub_M), 1./p)
            cost += proba_couple_batches * wass
    return cost


def get_div_mbot_with_r(a, b, all_arr_s, all_arr_t, bs_s, bs_t, M, M_s, M_t, p):
    '''Compute the divergence MBOT with replacement between 2 measures
    
       Parameters:
       -------------------------------------------
        a : ndarray, shape (ns,)
            Source measure.
        b : ndarray, shape (nt,)
            Target measure.
        all_arr_s : ndarray, shape (ns**m,m)
                    Source minibatch with replacement
        all_arr_t : ndarray, shape (nt**m,m)
                    Target minibatch with replacement
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
        value of MBOT divergence with replacement 
    '''

    OT_ab = get_mbot_with_r(a, b, all_arr_s, all_arr_t, bs_s, bs_t, M, p)
    OT_s = get_mbot_with_r(a, a, all_arr_s, all_arr_s, bs_s, bs_s, M_s, p)
    OT_t = get_mbot_with_r(b, b, all_arr_t, all_arr_t, bs_t, bs_t, M_t, p)
    return OT_ab - 1./2 * (OT_s + OT_t)
