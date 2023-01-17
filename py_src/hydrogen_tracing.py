from ase.neighborlist import natural_cutoffs, NeighborList
from ase.io import read as ase_read
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import re

from config_plot import configPlotter

class HOchainFinder():
    def __init__(self, trajectory, cutoff_mult=0.75):
        self.standard_config = 'H2O' # Sometimes needed to initialise configuration dictionary
        self.ti_regex = re.compile(r'Ti[2-9]')
        self.cutoff_mult = cutoff_mult
        self.trajectory = trajectory
        
        self.current_neighbourlist = None
        self.previous_neighbourlist = None
        
        # This is 
        self.skip_flag = False

        self._init_neighbourlists()
        
    def _init_neighbourlists(self):
        self.cut_offs = natural_cutoffs(self.trajectory[0], mult=self.cutoff_mult)
        self.current_neighbourlist = NeighborList(self.cut_offs, bothways=True, self_interaction=False)
        self.previous_neighbourlist = NeighborList(self.cut_offs, bothways=True, self_interaction=False)
        
    def _is_single_ti_bond(self, ii_snap, oxygen_index, **kwargs):
        neighbours = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
        symbols = self.trajectory[ii_snap][neighbours].get_chemical_formula(mode='hill')
        return ((re.search(self.ti_regex, symbols) is None) and not ("H2" in symbols))
    
    def _is_ti_bonded_and_changed(self, ii_snap, oxygen_index):
        neighbours = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
        symbols = self.trajectory[ii_snap][neighbours].get_chemical_formula(mode='hill')
        
        if 'Ti' in symbols and not "H2" in symbols: # Bonded to Ti 
            prev_neighbours = self.previous_neighbourlist.get_neighbors(oxygen_index)[0]
            diffs = np.setxor1d(neighbours, prev_neighbours)
            # Something changed in its neighbourhood
            if len(diffs) != 0:
                diff_symbols = self.trajectory[ii_snap][diffs].get_chemical_formula(mode='hill')
                # Check if H was involved in change
                if 'H' in diff_symbols:
                    return True
            
        return False
    
    def find_special_configs(self, is_special=None):
        # Return all special configurations in list of list of np.ndarrays
        # Starting with snapshot 1
        
        if is_special is None:
            is_special = self._is_ti_bonded_and_changed
        
        initial_config = self.trajectory[0]
        initial_config.set_pbc([True, True, True])
        
        special_list = []
        for ii_snapshot in range(1, len(self.trajectory)):
            if ii_snapshot % 100 == 0:
                print(ii_snapshot)
            # TODO: does every snapshot need this:
            # snapshot.set_pbc([True, True, True])
            snapshot = self.trajectory[ii_snapshot]
            self.current_neighbourlist.update(snapshot)
            self.previous_neighbourlist.update(self.trajectory[ii_snapshot-1])
            
            # TODO: Build indices once, then check if all oxygen indices stay oxygen
            oxygen_indices = [atom.index for atom in snapshot if atom.symbol == 'O']
            
            specials_count = 0
            special_sites = []
            for ii_index, oxygen_index in enumerate(oxygen_indices):
                # Use special picker to determine if oxygen is in analysable configuration
                if is_special(ii_snapshot, oxygen_index):
                    neighbours = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
                    specials_count += 1
                    special_sites.append(np.append(neighbours, oxygen_index).tolist())
            
            special_list.append(special_sites)
            # print(ii_snap, ": %u"%specials_count)        
            
        return special_list
    
    def all_configs(self):
        """Find all configurations of oxygen for all timesteps.
        Build an array of zeros shaped (n_timesteps, 1) and an empty config dict.
        Initialise config dict with some trial config:
        config_dict['H20'] == 0,
        which means 'H20' has entry 0 in count array.
        
        When a new config is found, append all zeros array to count array and make new entry in config_dict
        """
        # TODO: Build indices once, then check if all oxygen indices stay oxygen
        oxygen_indices = [atom.index for atom in self.trajectory[0] if atom.symbol == 'O']

        config_dict = {self.standard_config: 0}
        count_array = np.zeros((len(self.trajectory), 1), dtype=np.int16)
        config_codes = np.zeros(shape=(len(self.trajectory), len(oxygen_indices)), dtype=np.int8)

        for ii_snapshot in range(0, len(self.trajectory)):
            if ii_snapshot % 100 == 0:
                print(ii_snapshot)
            snapshot = self.trajectory[ii_snapshot]
            self.current_neighbourlist.update(snapshot)

            for ii_index, oxygen_index in enumerate(oxygen_indices):
                # Go through each oxygen index
                neighs = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
                ox_config = snapshot[np.append(neighs, oxygen_index)].get_chemical_formula(mode='hill')

                # TODO: make this a class method
                try: # key is already in config_dict
                    loc = config_dict[ox_config]
                except: # key in not in config_dict, add
                    loc = len(config_dict)
                    config_dict[ox_config] = loc
                    count_array = np.append(count_array, np.zeros((len(self.trajectory), 1), dtype=np.int16), axis=-1)
               
                count_array[ii_snapshot, loc] += 1
                config_codes[ii_snapshot, ii_index] = loc

        return config_dict, count_array, config_codes

    def _h30_processor(self, config, n_list):
        '''Fix configuration if neighbouring H is accidentally too close to O
        Alternative: Remove H from config that has two Os as neighbours
        
        config must be sorted according to distance to O
        '''
        atom = self.trajectory[0][config]
        if atom.get_chemical_formula(mode='hill') == 'H3O':
            symbols = atom.get_chemical_symbols()
            for ii_elem in range(len(symbols)):
                # First evaluate symbol H, then if it has more than one (Oxygen) neighbour, remove
                if symbols[ii_elem] == 'H' and len(n_list.get_neighbors(config[ii_elem])[0]) > 1:
                    config = np.delete(config, ii_elem)
                    break
        return config
            
    def find_hopping(self, oxygen_index, cur_step, counter, prev_step=None, verbose=False):
        if prev_step is None:
            prev_step = cur_step-1
            
        self.current_neighbourlist.update(self.trajectory[cur_step])
        self.previous_neighbourlist.update(self.trajectory[prev_step])
        
        special_config = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
        special_config = np.append(special_config, oxygen_index)
        # special_config = self._h30_processor(special_config, self.current_neighbourlist)
        prev_neighbours = self.previous_neighbourlist.get_neighbors(oxygen_index)[0]
        prev_config = np.append(prev_neighbours, oxygen_index)
        # prev_config = self._h30_processor(special_config, self.previous_neighbourlist)
        
        which_are_gone = np.setdiff1d(prev_config, special_config)
        which_are_added = np.setdiff1d(special_config, prev_config)
        
        cur_form = self.trajectory[cur_step][special_config].get_chemical_formula(mode='hill')
        prev_form = self.trajectory[cur_step][prev_config].get_chemical_formula(mode='hill')
        
        # Cases:
        # HOTiX which HO have bonded to: Either H2OTi and H has left or H20 and H has left
        # or whole H20 is double bonded 
        # OTiX which has been left by H: Problem for later
        # 
        # if in loop H20 or H20Ti which got an additional H and hopfully lost one H
        
        # First check if it's a OTiX where an H has left
        if counter == 0:
            if (('OTi' in cur_form) and not ('H' in cur_form)):
                # Simply ignore OTiX
                # If it shouldn't be ignored, just comment return statement
                return -1
                self.skip_flag = True
    
            if ('HOTi' in cur_form) and (('OTi' in prev_form) and not ('H' in prev_form)):
                # Simply ignore OTiX
                self.current_neighbourlist.update(self.trajectory[cur_step+1])
                neighs = self.current_neighbourlist.get_neighbors(oxygen_index)[0]
                atom = self.trajectory[cur_step][np.append(neighs, oxygen_index)]

                future_form = atom.get_chemical_formula(mode='hill')
                if 'HOTi' in future_form:
                    pass 
                    # self.counter += 1
                
                self.current_neighbourlist.update(self.trajectory[cur_step])
                return -2
            
        if ("H2O" in cur_form) and ('HOTi' in prev_form):
            # Special case where path is H20 -> HOTi + H but on other site HOTi + H -> H20
            if not self.skip_flag:
                return counter-1

        if verbose: # len(which_are_gone):
            print("Gone: ")
            print(self.trajectory[0][which_are_gone].get_chemical_formula())
        
        # Now it should be HOTiX
        if len(which_are_gone) != 1:
            # Optimally this should only be an H atom

            if len(which_are_gone) != 0:
                # If not, remove anything that is not H
                buff_gone = []
                ts_0 = self.trajectory[0]
                for gone in which_are_gone:
                    if ts_0[gone].symbol == 'H':
                        buff_gone.append(gone)
                which_are_gone = buff_gone

            if len(which_are_gone) == 0:
                # For some reason: no change, usually flyby
                return -6
        
        # Now length is definitely one
        if self.trajectory[0][which_are_gone].get_chemical_symbols()[0] != 'H':
            # print("Flyby")
            # print(self.trajectory[0][which_are_gone].get_chemical_symbols()[0])
            # For some reason: no change, usually flyby
            return -6

            
        prev_bound_oxygen = self.previous_neighbourlist.get_neighbors(which_are_gone[0])[0]
        if len(prev_bound_oxygen) > 1 and (counter == 0):
            gone_symbs = self.trajectory[0][prev_bound_oxygen].get_chemical_formula()
            if len(prev_bound_oxygen) > 2 or gone_symbs!='O2':
                return -3

            prev_bound_oxygen = np.setdiff1d(prev_bound_oxygen, oxygen_index)
        
        if verbose:
            print("####################")
            print("Steps")
            print(cur_step)
            print(prev_step)
            print("Configs")
            print(special_config, prev_config)
            print(self.trajectory[0][special_config].get_chemical_symbols(), self.trajectory[0][prev_config].get_chemical_symbols())
            print(which_are_gone)
            print(which_are_added)
            print("Forms")
            print(cur_form, prev_form)
            print("Gone: ", which_are_gone)
            print("Oxygen")
        
        # Find O that H is bond to now
        cur_gone_bound_oxygen = self.current_neighbourlist.get_neighbors(which_are_gone[0])[0]
        cur_bound = self.trajectory[cur_step][cur_gone_bound_oxygen].get_chemical_formula(mode='hill')
        if cur_bound != 'O':
            # H20 close to TiO, sharing bond
            if not (cur_bound == 'O2'):
                # This should never happen
                return -4 # fixed
            if verbose:
                print("Moved to 2 step")
            return self.find_hopping(oxygen_index, cur_step+1, counter, prev_step, verbose=verbose)
        
        # Find all neighbours of that oxygen
        cur_oxy_bond_inds = np.append(
            self.current_neighbourlist.get_neighbors(cur_gone_bound_oxygen[0])[0],
            cur_gone_bound_oxygen[0]
        )
        
        if verbose:
            print(self.previous_neighbourlist.get_neighbors(which_are_gone[0])[0])
            print(cur_gone_bound_oxygen)
            print(cur_bound)
            
        # cur_oxy_bond_inds = self._h30_processor(cur_oxy_bond_inds, self.current_neighbourlist)
        cur_oxy_bond = self.trajectory[cur_step][cur_oxy_bond_inds]
        cur_oxy_symbols = cur_oxy_bond.get_chemical_formula(mode='hill')
        # print(cur_oxy_symbols)
        if 'HOTi' in cur_oxy_symbols:
            self.skip_flag = False
            return counter
        elif 'H2O' in cur_oxy_symbols:
            if counter < 100:
                return self.find_hopping(cur_gone_bound_oxygen[0], cur_step, counter=counter+1, prev_step=prev_step, verbose=verbose)
            else:
                print(-9, cur_step, prev_step)
                return -9 # fixed
        else:
            # Probably H30, see how it develops over steps
            self.current_neighbourlist.update(self.trajectory[cur_step+1])
            if not self._is_ti_bonded_and_changed(cur_step+1, oxygen_index):
                # Isn't special site after all
                return -3
            return self.find_hopping(oxygen_index, cur_step+1, counter=counter+1, prev_step=prev_step, verbose=verbose)

    def analyse_special_configs(self, special_list, run_special: int=None):
        hop_list = []
        if run_special is None:
            for ii_step, specials in enumerate(special_list):
                # Go through all found special configurations
                ii_step += 1
                if ii_step % 100 == 0:
                    print(ii_step)
                
                if len(special_list) != 0:
                    for special_config in specials:
                        hops = self.find_hopping(special_config[-1], ii_step, counter=0, verbose=False)
                        hop_list.append(hops)
        else:
            for ii_step, specials in zip([run_special], [special_list[run_special]]):
                if len(special_list) != 0:
                    for special_config in specials:
                        hops = self.find_hopping(special_config[-1], ii_step, counter=0, verbose=True)
                        hop_list.append(hops)

        return np.asarray(hop_list)        

    def plot_special_config(self, snapshot, cut_offs=None):
        if cut_offs is None:
            cut_offs = self.cut_offs
            
        neighbour_list = NeighborList(cut_offs, bothways=True, self_interaction=False)
        neighbour_list.update(snapshot)
        oxygen_indices = [atom.index for atom in snapshot if atom.symbol == 'O']
        oxygen_neighbours = [
            neighbour_list.get_neighbors(oxygen_index)[0] for oxygen_index in oxygen_indices
        ]
        
        index_list = []
        neigh_list = []
        for ii_neighs, neighbours in enumerate(oxygen_neighbours):
            symbols = snapshot[neighbours].get_chemical_formula(mode='hill')
            if re.search(self.ti_regex, symbols) is None and not ("H2" in symbols):
                found_index = oxygen_indices[ii_neighs]
                neighs = neighbour_list.get_neighbors(found_index)[0]
                index_list.append(found_index)
                neigh_list.append(neighs)
        
        configPlotter.plot_snapshot(snapshot, index_list, neigh_list)