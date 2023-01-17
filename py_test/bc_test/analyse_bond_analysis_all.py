import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.neighborlist import NeighborList
from scipy.spatial.distance import cdist
from tqdm import tqdm
from matplotlib import cm
from matplotlib.patches import Patch

from asaplib.reducedim import Dimension_Reducers
from asaplib.plot import Plotters as asapPlotters

from ase.neighborlist import natural_cutoffs, NeighborList
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
        elif hh[2] > 2: # OH
            if  hh[3] < 1.5:
                cls[i] = 1 # H on O(TiO2)
            else:
                cls[i] = 2 # OH on Ti
        elif hh[5]<3:
            cls[i] = 3 # H2O on Ti
        else:
            cls[i] = 4 # H2O close to slab but not on Ti
            
    cls =  np.reshape( cls,(-1,nhydrogen))
    np.shape(cls)

    trajectory = read(os.path.join(this_dir, system_now+'/out.lammpstrj'), index='%u:%u:%u'%(ts_slicer.start, ts_slicer.stop, ts_slicer.step))
    cutoffs = natural_cutoffs(trajectory[0], mult=0.75)

    cur_nlist = NeighborList(cutoffs, bothways=True, self_interaction=False)
    prev_nlist = NeighborList(cutoffs, bothways=True, self_interaction=False)

    def get_o_config(nlist, h_index, init_config):
        h_neighs = nlist.get_neighbors(h_index)[0]
        
        if len(h_neighs) == 0:
            return 'solo'
        
        h_neigh = h_neighs[0] # Closest element should be O anyways
        
        o_neighs = nlist.get_neighbors(h_neigh)[0]
        o_neighs = np.append(h_neigh, o_neighs)
        
        return init_config[o_neighs].get_chemical_formula(mode='hill')

    # weighted transition rates (assuming V is quasi-static)

    start_configs = []
    end_configs = []
    wstart = []
    wend = []

    cl_transition = np.zeros((5,5))
    print(cls.shape)
    # I want to plot the evolution of H

    count = 0
    for i in range(1,np.shape(cls)[0]): # loop through the frames
        cur_nlist.update(trajectory[i])
        prev_nlist.update(trajectory[i-1])
        
        for j in range(nhydrogen):# loop through points
            [c1, c2] = [int(cls[i,j]), int(cls[i-1,j])]
            cl_transition[c1,c2] += 1 # h_weights[i,j]/h_weights[i-1,j]
            if (c1 == 1) and (c2 == 4):
                count += 1
                
                # print(f"Timestep: {1000+(10*i):n}")
                # print(f"Hydro Index: {h_dis_all[i, j, 0]:n}")
                # if count == 50:
                #     raise ValueError
                h_index = int(h_dis_all[i, j, 0])
                end_configs.append(get_o_config(cur_nlist, h_index, trajectory[0]))
                start_configs.append(get_o_config(prev_nlist, h_index, trajectory[0]))
            
            elif False: # c1 == 0 and c2 == 0:
                h_index = int(h_dis_all[i, j, 0])
                wend.append(get_o_config(cur_nlist, h_index, trajectory[0]))
                wstart.append(get_o_config(prev_nlist, h_index, trajectory[0]))
                
                
    print(count)
    print(cls.shape)
    print(cl_transition)

    for k in range(5):
        cl_norm = np.sum(cl_transition[k,:])
        cl_transition[k,:]/=cl_norm

    print("Starting configurations:")
    start_counts = pd.Series(start_configs).value_counts()
    print(start_counts)

    print("Final configurations:")
    end_counts = pd.Series(end_configs).value_counts()
    print(end_counts)

    print("Starting configurations:")
    counts = pd.Series(wstart).value_counts()
    print(counts)

    print("Final configurations:")
    counts = pd.Series(wend).value_counts()
    print(counts)

    fig, axes = plt.subplots(nrows=2, ncols=1)

    total_loc = -2
    x_start = np.arange(len(start_counts))
    x_end = np.arange(len(end_counts))
    fs = 15

    axes[0].bar(x_start, start_counts.array)
    axes[0].bar(total_loc, np.sum(start_counts.array))
    axes[0].set_xticks(np.append(total_loc, x_start))
    axes[0].set_xticklabels([r'H$_2$O$^{(1)}$'] + start_counts.keys().tolist())
    axes[0].annotate(
        "", 
        xy=(total_loc+0.5+1, np.sum(start_counts.array)/2), 
        xytext=(total_loc+0.5, np.sum(start_counts.array)/2), 
        arrowprops=dict(arrowstyle="->")
    )
    axes[0].set_title("Starting Configurations", fontsize=fs-5)
    axes[0].set_ylabel("counts")

    axes[1].bar(x_end, end_counts.array)
    axes[1].bar(total_loc, np.sum(end_counts.array))
    axes[1].set_xticks(np.append(total_loc, x_end))
    axes[1].set_xticklabels(['H-O$_t$'] + end_counts.keys().tolist())
    axes[1].annotate(
        "", 
        xy=(total_loc+0.5+1, np.sum(end_counts.array)/2), 
        xytext=(total_loc+0.5, np.sum(end_counts.array)/2), 
        arrowprops=dict(arrowstyle="->")
    )
    axes[1].set_title("End Configurations", fontsize=fs-5)
    axes[1].set_ylabel("counts")

    legend_elements = [
        Patch(facecolor='tab:orange', label='atomic descriptors classification'),
        Patch(facecolor='tab:blue', label='nearest neighbour environment')
    ]
    axes[0].legend(handles=legend_elements)

    fig.suptitle(f'{system_now.split("-")[0]} ({system_now.split("-")[1]})', fontsize=fs)

    plt.tight_layout()

    fig.savefig(os.path.join(this_dir, f'advsnl_{system_now.split("-")[0]}_{system_now.split("-")[1]}.pdf'), format='pdf')

    # plt.show()

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

    for system_now in system_list:
        run_on_file(system_now=system_now)

if __name__ == '__main__':
    main()