#Test file for sampling with replacement

import numpy as np
import ot
from utils import all_arr_w_replacement
from mbot_proba import get_div2
import matplotlib.pyplot as plt

###################################
# MAIN
###################################
#%% parameters and data generation

np.random.seed(1980)

ns = 4 # nb samples
nt = ns
m_s = 2
m_t = 2
dev_list = np.linspace(0, np.pi, 100)

# uniform distribution on samples
a, b = ot.unif(ns), ot.unif(nt)

#Test arrangements
all_arr_s = all_arr_w_replacement(ns, m_s)
all_arr_t = all_arr_w_replacement(nt, m_t)

total = 0
for arr_s in all_arr_s:
    for arr_t in all_arr_t:
        total += np.prod(a[arr_s]) * np.prod(b[arr_t])
print("SANITY PROBA CHECK : ", total)

w_value = np.zeros((1, len(dev_list)))

for id_dev, dev in enumerate(dev_list):
    xs = []
    xt = []
    for i in range(ns):
        xs.append([np.cos(2 * i * np.pi/ns), np.sin(2 * i * np.pi/ns)])
        xt.append([np.cos(2 * i * np.pi/ns + dev), np.sin(2 * i * np.pi/ns + dev)])
    xs = np.array(xs)
    xt = np.array(xt)

    # Cost matrix
    p = 1
    M = ot.dist(xs, xt, metric='euclidean')**p
    M_s = ot.dist(xs, xs, metric='euclidean')**p
    M_t = ot.dist(xt, xt, metric='euclidean')**p

    ###########################
    # TESTS
    ###########################

    divergence = get_div2(a, b, all_arr_s, all_arr_t, m_s, m_t, M, M_s, M_t, p)

    print("Expected div mini batch Wasserstein distance : ", divergence)

    w_value[0][id_dev] = divergence

plt.plot(dev_list, w_value[0])
plt.title('w value')
plt.xlabel('perturbation value')
plt.ylabel('loss value')
plt.tight_layout()
plt.show()
