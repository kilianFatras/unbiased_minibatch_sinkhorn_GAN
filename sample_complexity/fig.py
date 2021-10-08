import numpy as np
import matplotlib.pylab as pl

pl.rcParams['pdf.fonttype'] = 42
pl.rcParams['ps.fonttype'] = 42

fig = pl.figure(figsize=(12,4))
ax = fig.add_subplot(1,2,1)
def plot_perf(nlist, err, color, label, errbar=False, perc=20):
    pl.loglog(nlist, err.mean(0), label=label, color=color)
    if errbar:
        pl.fill_between(nlist, np.percentile(err,perc,axis=0), np.percentile(err,100-perc,axis=0),
                    alpha=0.2, facecolor=color)
        
fs = 16
list_d = [2, 5, 7]
list_m = [128] 
list_n = [1000*i+300 for i in range(11)]
values_dim = np.load('samples_comp.npy')
values_m = np.load('samples_comp_d_fix.npy')

 
plot_perf(list_n, values_dim[0, 0], 'g', 'd=2', errbar=True, perc=20)
plot_perf(list_n, values_dim[1, 0], 'b', 'd=7', errbar=True, perc=20)
plot_perf(list_n, values_dim[2, 0], 'r', 'd=10', errbar=True, perc=20)
pl.title(r'Sample complexity of $\widetilde{\Lambda}_{\overline{W_2^2}^\mathtt{W}}^k$, $k=10^3$, m=128', fontsize=fs)
pl.ylabel(r'value $\widetilde{\Lambda}_{\overline{W_2^2}^\mathtt{W}}^k(\alpha_n, \beta_n)$', fontsize=fs-1)
pl.xlabel('n : number of data', fontsize=fs-1)
pl.legend()
pl.tight_layout()

ax = fig.add_subplot(1,2,2)
plot_perf(list_n, values_m[0, 0], 'g', 'm=64', errbar=True, perc=20)
plot_perf(list_n, values_m[0, 1], 'b', 'm=128', errbar=True, perc=20)
plot_perf(list_n, values_m[0, 2], 'r', 'm=256', errbar=True, perc=20)
pl.title(r'Sample complexity of $\widetilde{\Lambda}_{\overline{W_2^2}^\mathtt{W}}^k$, $k=10^3$, d=7', fontsize=fs)
pl.ylabel(r'value $\widetilde{\Lambda}_{\overline{W_2^2}^\mathtt{W}}^k(\alpha_n, \beta_n)$', fontsize=fs-1)
pl.xlabel('n : number of data', fontsize=fs-1)
pl.legend()
pl.tight_layout()
pl.savefig('imgs/sample_complexity.pdf')
pl.show()
