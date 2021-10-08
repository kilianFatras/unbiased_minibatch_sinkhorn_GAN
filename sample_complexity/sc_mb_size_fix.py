#%% import
import numpy as np
import matplotlib.pylab as pl
import ot
from utils import comb_index
from utils import plot_OT_mat, imshow_OT, plot_norm_error
from mini_batch_ot import calculate_inc_mbot



def plot_perf(nlist, err, color, label, errbar=False, perc=20):
    pl.loglog(nlist, err.mean(0), label=label, color=color)
    if errbar:
        pl.fill_between(nlist, np.percentile(err,perc,axis=0), np.percentile(err,100-perc,axis=0),
                    alpha=0.2, facecolor=color)

###################################
# MAIN
###################################
#%% parameters and data generation
np.random.seed(1985)
k = 1000

########################################
# Sample complexity for a fix mb size
########################################
list_d = [2, 7, 10]
list_m = [128] 
list_n = [1000*i+300 for i in range(11)]
print(list_n)
n_exp = 5
values = np.zeros((len(list_d), len(list_m), n_exp, len(list_n)))
print(max(list_n))

for i_d, d in enumerate(list_d):
    for i_m, m in enumerate(list_m):
        for id_exp in range(n_exp):
            for i_n, ns in enumerate(list_n):
                print(d, m, id_exp, ns)
                gamma = np.zeros((ns, ns))
                nt = ns

                xs = np.random.uniform(0, 1, (ns, d)) #np.random.rand(nt, d)
                xt = np.random.uniform(0, 1, (nt, d))

                a, b = ot.unif(ns), ot.unif(nt)  # uniform distribution on samples

                #Value div MBOT without replacement
                M = ot.dist(xs, xt)
                cost_ab = calculate_inc_mbot(xs, xt, a, b, m, m, k, M)
                
                Ms = ot.dist(xs, xs)
                cost_aa = calculate_inc_mbot(xs, xs, a, a, m, m, k, Ms)
                
                Mt = ot.dist(xt, xt)
                cost_bb = calculate_inc_mbot(xt, xt, b, b, m, m, k, Mt)
                values[i_d, i_m, id_exp, i_n] = np.abs(cost_ab - 1/2*cost_aa - 1/2*cost_bb)

###########################################################################
##############                  PLOTS                       ###############
###########################################################################

plot_perf(list_n, values[0, 0], 'g', 'd=2', errbar=True, perc=20)
plot_perf(list_n, values[1, 0], 'b', 'd=7', errbar=True, perc=20)
plot_perf(list_n, values[2, 0], 'r', 'd=10', errbar=True, perc=20)
pl.title('Sample complexity of div U, k=10^3, m=128')
pl.ylabel('value div U(a_n, b_n)')
pl.xlabel('n : number of data')
pl.legend()
pl.tight_layout()
pl.savefig('imgs/sample_comp_UD_1k')
pl.show()
np.save('samples_comp.npy', values)

