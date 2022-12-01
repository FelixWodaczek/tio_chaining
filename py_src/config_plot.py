from ase.neighborlist import natural_cutoffs
import matplotlib.pyplot as plt
import numpy as np

class configPlotter():
    @staticmethod
    def _mark_masses(masses, snapshot, found_index, neighs):
        found_index = np.array([found_index])
        mark_indices = np.append(neighs, found_index, axis=0)
        masses[mark_indices] = 100
        return masses
        
    @staticmethod
    def plot_snapshot(snapshot, found_index, neighs):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection='3d')
        
        at_pos = snapshot.get_positions()
        masses = snapshot.get_masses()

        if isinstance(found_index, list):
            for single_index, single_neighs in zip(found_index, neighs):
                configPlotter._mark_masses(masses, snapshot, single_index, single_neighs)
        elif isinstance(found_index, int):
            configPlotter._mark_masses(masses, snapshot, found_index, neighs)

        sizes = np.array(natural_cutoffs(snapshot, mult=500))
        
        sc = ax.scatter(
            at_pos[:, 0], at_pos[:, 1], at_pos[:, 2], 
            s=sizes,
            c=masses,
            alpha=1,
            edgecolors="k", # vmin=0, vmax=1
        )
        
        z_span = (ax.get_zlim()[1] - ax.get_zlim()[0])/2.
        ax.set_ylim([np.mean(ax.get_ylim()) - z_span, np.mean(ax.get_ylim()) + z_span])
        ax.set_xlim([np.mean(ax.get_xlim()) - z_span, np.mean(ax.get_xlim()) + z_span])
        # cbar = fig.colorbar(sc)
        plt.show()