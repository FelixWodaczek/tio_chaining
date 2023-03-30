# README

Small README file for SI in publication [Water dissociation on pristine low-index TiO2 surfaces](https://doi.org/10.48550/arXiv.2303.07433).
This repository contains, albeit not very cleanly, the code for hydrogen tracing using neighbor lists.
Source files are found in `py_src`, the running scripts are contained within `py_test`.
For data feel free to contact the owner of the repository.

## Overview

The source code is very brief and built upon tracing the entering of H atoms through neighborhoods of O atoms.
For a script showing the workings of this algorithm see `py_test/chain_finding.ipynb`.
In general, for a trajectory, the code works in the following manner:
```python
import hydrogen_tracing
from ase.io import read as ase_read

file_path = '<insert_path_here>'
trajectory = ase_read(file_path) # Load the trajectory using ASE Library

# Create a chain finder class to work on trajectory
chain_finder = hydrogen_tracing.HOchainFinder(trajectory) 

# Not necessary for algorithm, but this gives all oxygen environments for each timestep
config_dict, counts, config_codes = chain_finder.all_configs()
configs = ['']*len(config_dict)
for key, value in config_dict.items():
    configs[value] = key

total_counts = np.sum(counts, axis=0)
print("In total found: ")
for ii_conf in range(total_counts.shape[0]):
    print(configs[ii_conf]+': ', total_counts[ii_conf])

# Now algorithm is run by first finding oxygens, where an H has left from previous timestep
special_list = chain_finder.find_special_configs()

# Determine what type of proton transfer has taken place in special_list
hop_list = chain_finder.analyse_special_configs(special_list, verbose=False)
```

## Publication Plots

The plots for publication were created using `py_test/plot_si.ipynb` using results calculated on a cluster.
To get these results, pull them from this repository via git-lfs, after which they can be found in `test_data/230302_short_traj_publike_all/`