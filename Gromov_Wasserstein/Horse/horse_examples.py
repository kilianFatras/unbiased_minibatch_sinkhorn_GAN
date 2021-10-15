import sys
import numpy as np
import torch
sys.path.append('./lib')
import pylab as plt
from matplotlib import gridspec
import imageio
import os

import ot
from mini_batch_gw import calculate_stoc_gamma2


path='./data/meshes.npy'
tab_obj=np.load(path)


fig = plt.figure(figsize=(15,10))

import re
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


print('start horse motion expe')
alldist_mbgw=[]
X1=tab_obj[0]
n_samples = len(X1)
m = 256
num_iter = 10
M1 = ot.dist(X1, X1)
M1 /= M1.max()
a = ot.unif(n_samples)
b = ot.unif(n_samples)

for i in range(len(tab_obj)):
    print(i)
    M2 = ot.dist(tab_obj[i], tab_obj[i])
    M2 /= M2.max()
    MBGW = calculate_stoc_gamma2(X1, tab_obj[i], a, b, m, m, num_iter, M1, M2) #  MBGW s->t
    
    #  Uncomment if you want to compute the divergence
    #MBGW -= 1/2*calculate_stoc_gamma2(X1, X1, a, a, m, m, num_iter, M1, M1) #  MBGW s->s
    #MBGW -= 1/2*calculate_stoc_gamma2(tab_obj[i], tab_obj[i], b, b, m, m, num_iter, M2, M2) #  MBGW t->t
    
    alldist_mbgw.append(MBGW)

print(max(alldist_mbgw))

for i in range(len(tab_obj)):
    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4,2]) 

    ax = fig.add_subplot(gs[0,0], projection='3d')

    X1=tab_obj[0]
    ax.scatter(X1[:,0],X1[:,2],X1[:,1], marker='o', s=20, c="green", alpha=0.6)
    ax.view_init(elev=10., azim=360)
    ax.set_ylim([-0.5,0.8])
    ax.set_xlim([-0.5,0.5])
    ax.set_zlim([-0.5,0.8])
    
    ax.set_axis_off()


    ax = fig.add_subplot(gs[0,1], projection='3d')
    X2=tab_obj[i]
    ax.scatter(X2[:,0],X2[:,2],X2[:,1], marker='o', s=20, c="goldenrod", alpha=0.6)
    ax.view_init(elev=10., azim=360)
    ax.set_ylim([-0.5,0.8])
    ax.set_xlim([-0.5,0.5])
    ax.set_zlim([-0.5,0.8])

    ax.set_axis_off()


    ax = fig.add_subplot(gs[1,:])
    plt.plot(alldist_mbgw[0:i],c='r',lw=2,marker='o')
    plt.ylim([0, max(alldist_mbgw)])
    plt.xlim([0,50])
    plt.legend(['MB Gromov-Wasserstein'])
    
    plt.tight_layout()

    plt.suptitle('MBGW on galopping horses with m={},k={}'.format(m, num_iter),fontsize=15)
    fig.savefig('./res/'+str(i)+'.png', dpi=fig.dpi)


path='./res/'
filenames=sorted_aphanumeric(os.listdir(path))
files=[path+f for f in filenames if 'png' in f]
images = []
for filename in files:
    images.append(imageio.imread(filename))
    
imageio.mimsave('horse.gif', images,fps=3)

