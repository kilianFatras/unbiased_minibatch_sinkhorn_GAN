import numpy as np
import matplotlib.pylab as pl
import ot
from utils import mini_batch, update_gamma
import scipy



def calculate_expectation_gamma(a, b, all_comb_s, all_comb_t, bs_s, bs_t, M1, M2,lambd=0.5, method='emd'):
    '''Compute the expectation of mini batch gamma'''
    nb_comb_s = np.shape(all_comb_s)[0]
    nb_comb_t = np.shape(all_comb_t)[0]
    gamma = np.zeros((np.shape(M1)[0], np.shape(M1)[1]))
    sum_weights = 0
    for i in range(nb_comb_s):
        id_a = all_comb_s[i]
        for j in range(nb_comb_t):
            id_b = all_comb_t[j]
            sub_weights_a = a[id_a]/np.sum(a[id_a])#ot.unif(bs_s)
            sub_weights_b = b[id_b]/np.sum(b[id_b])#ot.unif(bs_t)

            if method == 'emd':
                sub_M1 = M1[id_a, :][:, id_a].copy()
                sub_M2 = M2[id_b, :][:, id_b].copy()
                G0 = ot.gromov.gromov_wasserstein(sub_M1, sub_M2, sub_weights_a, sub_weights_b, 'square_loss')

            elif method == 'entropic':
                sub_M1 = M1[id_a, :][:, id_a]
                sub_M2 = M2[id_b, :][:, id_b]
                G0 = ot.gromov.entropic_gromov_wasserstein(sub_M1, sub_M2, sub_weights_a, sub_weights_b,
                                                           'square_loss', epsilon=lambd)

            #Test update gamma
            cur_weight = np.sum(a[id_a]) * np.sum(b[id_b])
            gamma = update_gamma(gamma, cur_weight * G0, id_a, id_b)
            sum_weights += cur_weight
    return (1/(sum_weights)) * gamma


def calculate_expectation_gamma2(a, b, all_comb_s, all_comb_t, bs_s, bs_t, M1, M2,lambd=0.5, method='emd'):
    '''Compute the expectation of mini batch gamma'''
    nb_comb_s = np.shape(all_comb_s)[0]
    nb_comb_t = np.shape(all_comb_t)[0]
    avg_gw_val = 0
    sum_weights = 0
    for i in range(nb_comb_s):
        id_a = all_comb_s[i]
        for j in range(nb_comb_t):
            id_b = all_comb_t[j]
            sub_weights_a = a[id_a]/np.sum(a[id_a])#ot.unif(bs_s)
            sub_weights_b = b[id_b]/np.sum(b[id_b])#ot.unif(bs_t)

            if method == 'emd':
                sub_M1 = M1[id_a, :][:, id_a].copy()
                sub_M2 = M2[id_b, :][:, id_b].copy()
                val_GW = ot.gromov.gromov_wasserstein2(sub_M1, sub_M2, sub_weights_a, sub_weights_b, 'square_loss')[0]

            elif method == 'entropic':
                sub_M1 = M1[id_a, :][:, id_a]
                sub_M2 = M2[id_b, :][:, id_b]
                val_GW = ot.gromov.entropic_gromov_wasserstein2(sub_M1, sub_M2, sub_weights_a, sub_weights_b, 'square_loss', epsilon=lambd)[0]

            #Test update gamma
            cur_weight = np.sum(a[id_a]) * np.sum(b[id_b])
            avg_gw_val += cur_weight * val_GW
            sum_weights += cur_weight
    return (1/(sum_weights)) * avg_gw_val


def calculate_stoc_gamma(xs, xt, a, b, bs_s, bs_t, num_iter, M1, M2,
                        lambd=1e-1, method='emd'):
    '''Compute the mini batch gamma with stochastic source and target'''
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
    '''Compute the mini batch gamma with stochastic source and target'''
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
                sub_weights_a, sub_weights_b, 'square_loss')[0]

        elif method == 'entropic':
            sub_M1 = M1[id_a, :][:, id_a]
            sub_M2 = M2[id_b, :][:, id_b]
            val_GW = ot.gromov.entropic_gromov_wasserstein2(sub_M1, sub_M2,
            sub_weights_a, sub_weights_b, 'square_loss', epsilon=lambd)[0]

        #Test update gamma
        cur_weight = np.sum(a[id_a]) * np.sum(b[id_b])
        avg_gw_val += cur_weight * val_GW
        sum_weights += cur_weight

    return (1/sum_weights) * avg_gw_val
