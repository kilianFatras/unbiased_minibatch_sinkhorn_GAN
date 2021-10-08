import numpy as np
import ot
from utils import all_arr_w_replacement, comb_index
from mbot_with_replacement import get_div_mbot_with_r
from mbot_without_replacement import get_div_mbot_without_r
import matplotlib.pyplot as plt

###################################
# MAIN
###################################
#%% parameters and data generation
if __name__ == '__main__':

    np.random.seed(1980)

    ns = 8 # nb samples
    nt = ns
    m_s = 2
    m_t = 2
    dev_list = np.linspace(0, np.pi, 100)

    # uniform distribution on samples
    a, b = ot.unif(ns), ot.unif(nt)
    list_p = [1, 2]
    values = np.zeros((len(list_p), 2, len(dev_list)))

    run = True
    if run:
        for id_p, p in enumerate(list_p):
            print("p : ", id_p)
            for id_exp in range(2):
                print("num exp : ", id_exp)
                if id_exp == 0:
                    all_mb_s = comb_index(ns, m_s)
                    all_mb_t = comb_index(nt, m_t)

                else:
                    all_mb_s = all_arr_w_replacement(ns, m_s)
                    all_mb_t = all_arr_w_replacement(nt, m_t)

                for id_dev, dev in enumerate(dev_list):
                    xs = []
                    xt = []
                    for i in range(ns):
                        xs.append([np.cos(2 * i * np.pi/ns), np.sin(2 * i * np.pi/ns)])
                        xt.append([np.cos(2 * i * np.pi/ns + dev), np.sin(2 * i * np.pi/ns + dev)])
                    xs = np.array(xs)
                    xt = np.array(xt)

                    # Cost matrix
                    M = ot.dist(xs, xt, metric='euclidean')**p
                    M_s = ot.dist(xs, xs, metric='euclidean')**p
                    M_t = ot.dist(xt, xt, metric='euclidean')**p
                    ###########################
                    # TESTS
                    ###########################

                    if id_exp == 1:
                        divergence = get_div_mbot_with_r(a, b, all_mb_s, all_mb_t, m_s, m_t,
                                                         M, M_s, M_t, p)
                        print("Div MBOT with r: ", divergence)

                    else:
                        divergence = get_div_mbot_without_r(a, b, all_mb_s, all_mb_t, m_s, m_t, 
                                                            M, M_s, M_t, p)

                        print("Div MBOT without r: ", divergence)

                    values[id_p][id_exp][id_dev] = divergence
        #np.save('all_values_mbot', values)
    else:
        values = np.load('all_values_mbot.npy')

    xs = []
    xt = []
    for i in range(ns):
        xs.append([np.cos(2 * i * np.pi/ns), np.sin(2 * i * np.pi/ns)])
        xt.append([np.cos(2 * i * np.pi/ns + 0.1), np.sin(2 * i * np.pi/ns + 0.1)])
    xs = np.array(xs)
    xt = np.array(xt)

    zeros = np.zeros(len(dev_list))

    fig = plt.figure(figsize=(14,4))
    ax1 = fig.add_subplot(1,3,1)

    plt.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    plt.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    plt.xticks([], []); plt.yticks([], [])
    plt.title('Source and target distributions', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()


    ax1 = fig.add_subplot(1,3,2)
    ax1.plot(dev_list, zeros, color='red')
    ax1.plot(dev_list, values[0][0], color='blue', label=r'$\Lambda_{W_1,w^\mathtt{W},P^\mathtt{W}, C(\mathbf{X}, \mathbf{Y})}$')
    #plt.plot(dev_list, values[2], color='green', label='V est.')
    ax1.plot(dev_list, values[0][1], color='orange', label=r'$\Lambda_{W_1,w^\mathtt{U},P^\mathtt{U}, C(\mathbf{X}, \mathbf{Y})}$')
    plt.xlabel('perturbation value', fontsize=14)
    plt.ylabel('loss value', fontsize=14)
    plt.title(r'$\Lambda_{W_1}$ value between 2D distributions', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()

    ax2 = fig.add_subplot(1,3,3)
    ax2.plot(dev_list, zeros, color='red')
    ax2.plot(dev_list, values[1][0], color='blue', label=r'$\Lambda_{W_2,w^\mathtt{W},P^\mathtt{W}, C(\mathbf{X}, \mathbf{Y})}$')
    ax2.plot(dev_list, values[1][1], color='orange', label=r'$\Lambda_{W_2 ,w^\mathtt{U},P^\mathtt{U}, C(\mathbf{X}, \mathbf{Y})}$')
    plt.xlabel('perturbation value', fontsize=14)
    plt.ylabel('loss value', fontsize=14)
    plt.title(r'$\Lambda_{W_2}$ value between 2D distributions', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()


    plt.savefig('comparison_p_{}.png'.format(1))
    plt.show()
