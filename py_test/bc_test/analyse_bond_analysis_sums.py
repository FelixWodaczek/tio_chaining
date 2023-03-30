import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from scipy.spatial.distance import cdist
from tqdm import tqdm
from matplotlib import cm
from matplotlib.patches import Patch
import json

from asaplib.reducedim import Dimension_Reducers
from asaplib.plot import Plotters as asapPlotters

import pandas as pd

def run_on_file(system_now):
    # Possible reshaping issue with 'rutile-100-nd-0', seems to not be 12 long but 11
    this_dir = os.path.dirname(os.path.abspath(__file__))

    nhydrogen = 128*2

    ts_slicer = np.s_[1000:10000:10]

    system_ref = 'anatase-101-nd-0' # always use this
    with open(os.path.join(this_dir, 'ref-anatase-101-nd-0-h-dis.npy'), 'rb') as f:
        h_dis_all_ref = np.load(f)
        print(h_dis_all_ref.shape)

    colvar = np.genfromtxt(os.path.join(this_dir, system_now+'/COLVAR'))[ts_slicer] # consistent with the gen of features
    tt = 330 #K
    kbt = 0.008314*tt # in kj/mol

    print(len(colvar))
    sys_weights = np.exp((colvar[:,2]+colvar[:,3])/kbt)

    h_weights = np.ones((len(sys_weights),nhydrogen))
    for i in range(len(sys_weights)):
        h_weights[i,:] *= sys_weights[i]

    h_weights = np.ones((len(sys_weights),nhydrogen))
    for i in range(len(sys_weights)):
        h_weights[i,:] *= sys_weights[i]

    with open(os.path.join(this_dir, system_now+'-h-dis-env.npy'), 'rb') as f:
        h_dis_all = np.load(f)
        h_env_all = np.load(f)
        print(h_dis_all.shape)
        print(h_env_all.shape)

    reduce_dict = {}
    """
    reduce_dict['pca'] = {
        "type": 'PCA',
        'parameter':{
            "n_components": 4}
    }
    """
    reduce_dict['kpca'] = {
        "type": 'SPARSE_KPCA',
        'parameter':{
            "n_components": 2,
            "n_sparse": 200, # no sparsification
            "kernel": {"first_kernel": {"type": 'cosine'}}
        }
    }

    dreducer = Dimension_Reducers(reduce_dict)

    hcoord_ref = np.reshape(h_dis_all_ref,(-1,11))
    hcoord_now = np.reshape(h_dis_all,(-1,h_dis_all.shape[-1]))

    print(hcoord_ref.shape)
    print(hcoord_now.shape)

    # proj = dreducer.fit_transform(hcoord_now[:,[1,2,3,5,6,8,9,10]])

    dreducer.fit(hcoord_ref[:,[1,2,3,5,6,8,9,10]])
    proj = dreducer.transform(hcoord_now[:,[1,2,3,5,6,8,9,10]])

    print(proj.shape)

    # reshape
    print(proj.shape[0]/nhydrogen)
    h_proj = np.reshape( proj[:,[0,1]],(-1,nhydrogen,2))
    print(np.shape(h_proj))

    stride = 1
    h_proj_sparse = h_proj[::stride,:,:]
    np.shape(h_proj_sparse)

    h_proj_sparse_all = np.reshape(h_proj_sparse[1:,:,:],(-1,2))

    # classify

    cls_labels= ['H$_2$O$^{(> 1)}$','H-O$_t$','HO-Ti','H$_2$O-Ti','H$_2$O$^{(1)}$']

    cls = np.zeros(len(hcoord_now))
    for i,hh in enumerate(hcoord_now):
        if hh[3] >=5: 
            cls[i] = 0 # in the bulk
        elif hh[3] < 1.2 and hh[2] > 1.6:
            cls[i] = 1 # H on O(TiO2)
        elif hh[2] > 1.8 and hh[5]<3: # OH on Ti
            cls[i] = 2 # OH on Ti
        elif hh[5]<3:
            cls[i] = 3 # H2O on Ti
        else:
            cls[i] = 4 # H2O close to slab but not on Ti
            
    cls =  np.reshape( cls,(-1,nhydrogen))

    # weighted transition rates (assuming V is quasi-static)


    cl_transition = np.zeros((5,5))
    print(cls.shape)
    # I want to plot the evolution of H

    count = 0
    for i in range(1,np.shape(cls)[0]): # loop through the frames
        for j in range(nhydrogen):# loop through points
            [c1, c2] = [int(cls[i,j]), int(cls[i-1,j])]
            cl_transition[c1,c2] += 1 # h_weights[i,j]/h_weights[i-1,j]
                
    return [int(cl_transition[1, 3]), int(cl_transition[1, 4])]


def main():
    system_list = [
        'anatase-100-nd-0',
        'anatase-101-nd-0',
        'anatase-110-nd-0',
        'rutile-001-nd-0',
        'rutile-011-nd-0',
        'rutile-100-nd-0',#'rutile-101-nd-0',
        'rutile-110-nd-0'
    ]
    
    count_dict = {}
    for system_now in system_list:
        count_dict[system_now] = run_on_file(system_now=system_now)
    
    with open("ad_counts.json", 'w') as f:
        json.dump(count_dict, f)
        f.close()

if __name__ == '__main__':
    main()