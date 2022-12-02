from ase.io import read as ase_read
import hydrogen_tracing 
import os
import numpy as np
import json

TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_data')

TARGET_PATHS = [
    "anatase-001-nd-0/out.lammpstrj",
    "anatase-100-nd-0/out.lammpstrj",
    "anatase-101-nd-0/out.lammpstrj",
    "anatase-110-nd-0/out.lammpstrj",
    "rutile-001-nd-0/out.lammpstrj",
    "rutile-011-nd-0/out.lammpstrj",
    "rutile-100-nd-0/out.lammpstrj",
    "rutile-101-nd-0/out.lammpstrj",
    "rutile-110-nd-0/out.lammpstrj",
]

def analyse_file(fpath, max_index: int=None):
    if max_index is None:
        trajectory = ase_read(fpath, index=':')
    else:
        trajectory = ase_read(fpath, index=':%u'%max_index)
    dirname = os.path.dirname(fpath)

    chain_finder = hydrogen_tracing.HOchainFinder(trajectory)
    config_dict, counts, config_codes = chain_finder.all_configs()
    special_list = chain_finder.find_special_configs()
    hop_list = chain_finder.analyse_special_configs(special_list)

    non_numpy_results = {
        'config_dict': config_dict,
        'special_list': special_list
    }
    
    with open(os.path.join(dirname, os.path.basename(dirname)+'_non_numpy.json'), 'w') as f:
        json.dump(non_numpy_results, f)
        f.close()
    
    np.savetxt(os.path.join(dirname, os.path.basename(dirname)+'_counts.txt'), counts)
    np.savetxt(os.path.join(dirname, os.path.basename(dirname)+'_config_codes.txt'), config_codes)
    np.savetxt(os.path.join(dirname, os.path.basename(dirname)+'_hop_list.txt'), hop_list)

if __name__ == "__main__":
    for path in TARGET_PATHS:
        full_path = os.path.join(TEST_DIR, path)
        dirname = os.path.dirname(full_path)
        analyse_file(full_path)