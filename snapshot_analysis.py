# reads snapshots of SWIFT simulations for plotting and analysis
import swiftsimio as sw
from swiftsimio.visualisation.slice import slice_gas
from swiftsimio.visualisation.projection import project_gas
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
import woma
from unyt import Rearth, Mearth, kg

import numpy as np


# class that stores and analyses particle data in a SWIFT snapshot
class snapshot:

    def __init__(self, filename):

        # loads particle data
        self.data = sw.load(filename)
        print(f'Loaded {len(self.data.gas.densities)} particles')

        self.box_size = self.data.gas.metadata.boxsize
        self.center_of_mass = self.get_center_of_mass()

        self.data.gas.masses.convert_to_mks()
        self.total_mass = np.sum(self.data.gas.masses)
        self.total_mass.convert_to_units(Mearth)
        print(f'Total mass of particles {self.total_mass:.4e}')

        # calculates the coordinates of the particles relative to the CoM
        pos = np.array(self.data.gas.coordinates - self.center_of_mass)
        self.R_xy = np.hypot(pos[:, 0], pos[:, 1]) * Rearth
        self.z = pos[:, 2] * Rearth
        self.r = np.hypot(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2]) * Rearth



    # calculates the centre of mass in the snapshot
    def get_center_of_mass(self):

        pos, densities = np.array(self.data.gas.coordinates), np.array(self.data.gas.densities)
        mass_sum, mass_pos_sum = 0, 0

        for i in range(pos.shape[0]):
            mass_pos_sum += pos[i] * densities[i]
            mass_sum += densities[i]
        center_of_mass = np.array(mass_pos_sum / mass_sum) * Rearth

        print(f'Center of mass found at {center_of_mass}')
        return center_of_mass

    # uses unyt
    def mass_within_r(self, r):

        if type(r) is np.ndarray:
            mass_total = np.zeros_like(r)
            for i in range(len(r)):
                mass_total[i] = np.sum(self.data.gas.masses[self.r < r[i]])
            result = mass_total * kg
        else:
            self.data.gas.masses.convert_to_mks()
            result = np.sum(self.data.gas.masses[self.r < r])

        result.convert_to_unit(Mearth)
        return result
