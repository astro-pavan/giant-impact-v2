# extracts data from a snapshot and analyses it, producing a 2D photosphere from a 3D grid
import sys

import numpy as np
from matplotlib import pyplot as plt
from swiftsimio.visualisation.volume_render import render_gas
from snapshot_analysis import snapshot

from unyt import Rearth


class photosphere:

    def __init__(self, snapshot, size, resolution=200):

        print('Loading snapshot data into grid...')

        center = snapshot.center_of_mass
        limits = [center[0] - size, center[0] + size,
                  center[1] - size, center[1] + size,
                  center[2] - size, center[2] + size]

        cell_size = (2 * size) / resolution

        self.vals = {'rho': render_gas(
            snapshot.data,
            resolution=resolution,
            project="masses",
            region=limits,
            parallel=True
        )}

        def get_cube(parameter):
            mass_weighted_cube = render_gas(
                snapshot.data,
                resolution=resolution,
                project=f'{parameter}_mass_weighted',
                region=limits,
                parallel=True
            )
            return mass_weighted_cube / self.vals['rho']

        self.vals['T'] = get_cube('temperatures')
        self.vals['P'] = get_cube('pressures')
        self.vals['s'] = get_cube('entropy')
        self.vals['u'] = get_cube("internal_energies")
        self.vals['h'] = get_cube("angular_momentum")

        print(f'Grid loaded')

        cross_P = np.log10(self.vals['P'][:, int(resolution/2), :])
        midplane_P = np.log10(self.vals['P'][:, :, int(resolution / 2)])

        cross_rho = np.log10(self.vals['rho'][:, int(resolution / 2), :])
        midplane_rho = np.log10(self.vals['rho'][:, :, int(resolution / 2)])

        fig, ax = plt.subplots()
        ax.imshow(midplane_rho, cmap='plasma')
        plt.show()

        fig, ax = plt.subplots()
        ax.imshow(cross_rho, cmap='plasma')
        plt.show()


snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
photosphere(snapshot1, 10 * Rearth, 200)

