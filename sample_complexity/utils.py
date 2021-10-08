import numpy as np
import matplotlib.pylab as pl


##############################################################
# Minibatch related functions
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
    id = np.random.choice(np.shape(data)[0], batch_size, replace=False)
    sub_weights = weights[id]/np.sum(weights[id])
    return data[id], sub_weights, id


##############################################################
# Plot functions
##############################################################

    
def plot_perf(nlist, err, color, label, errbar=False, perc=20):
    pl.loglog(nlist, err.mean(0), label=label, color=color)
    if errbar:
        pl.fill_between(nlist, np.percentile(err,perc,axis=0), np.percentile(err,100-perc,axis=0),
                    alpha=0.2, facecolor=color)