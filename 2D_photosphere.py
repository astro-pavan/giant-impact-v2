# extracts data from a snapshot and analyses it, producing a 2D photosphere model

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from scipy.interpolate import interp1d
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import cm, g, Rearth, J, K, kg
from tqdm import tqdm

from snapshot_analysis import snapshot


class photosphere:

    def __init__(self, snapshot, size, resolution):

        self.data = {}
        self.snapshot = snapshot

        # used to convert coordinates to indexes in the slices
        pixels_per_Rearth = resolution / float(2 * size)
        self.cell_size = (size * 2) / resolution
        self.size, self.resolution = size, resolution

        # loads a cross-section of the snapshot
        def get_section(phi, fold=True):
            # properties used to load the slices
            center = snapshot.center_of_mass
            rotate_z = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='z')
            rotate_x = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='x')
            matrix = np.matmul(rotate_x, rotate_z)
            limits = [center[0] - size, center[0] + size, center[1] - size, center[1] + size]

            # loads density slice
            mass_slice = slice_gas(snapshot.data,
                z_slice=0,
                resolution=resolution,
                project="masses",
                region=limits, rotation_matrix=matrix, rotation_center=center,
                parallel=True
            )

            # function that loads the slice of each property
            def get_slice(parameter):
                mass_weighted_slice = slice_gas(
                    snapshot.data,
                    z_slice=0,
                    resolution=resolution,
                    project=f'{parameter}_mass_weighted',
                    region=limits, rotation_matrix=matrix, rotation_center=center,
                    parallel=True
                )

                return mass_weighted_slice / mass_slice

            # loading slices of each property
            temperature_slice = get_slice('temperatures')
            pressure_slice, entropy_slice = get_slice('pressures'), get_slice('entropy')
            angular_momentum_slice = get_slice('angular_momentum')

            mass_slice.convert_to_units(g / cm ** 3)

            data = {'rho': mass_slice, 'T': temperature_slice, 'P': pressure_slice, 's': entropy_slice,
                    'h': angular_momentum_slice}

            if fold:
                for k in data.keys():
                    data[k] = (data[k] + np.fliplr(data[k])) / 2

            return data

        # loads multiple sections at different phi angles and averages them
        def get_averaged_data(n):

            data = get_section(0)

            print('Loading data into photosphere model:')
            for i in tqdm(range(1, n)):
                vals = get_section(np.pi/n * i)
                for k in self.data.keys():
                    data[k] = (i * data[k] + vals[k]) / (i + 1)

            return data

        # loads phi averaged data
        self.data = get_averaged_data(1)

        # removes half the data array
        for k in self.data.keys():
            self.data[k] = self.data[k][:,int(resolution/2):]

        # extends the data array
        extend = resolution
        for k in self.data.keys():
            unit = self.data[k].units
            self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend)), 'mean' if k =='z' else 'constant') * unit

        # calculating coordinates
        rows, cols = self.data['rho'].shape[0], self.data['rho'].shape[1]
        iX, iY = np.meshgrid(np.arange(cols), np.arange(rows))
        self.data['R'] = ((iX + 0.5) / pixels_per_Rearth) * Rearth
        self.data['z'] = ((iY - resolution / 2 + 0.5) / pixels_per_Rearth) * Rearth
        self.data['r'] = np.sqrt(self.data['R'] ** 2 + self.data['z'] ** 2)
        self.data['theta'] = np.arctan2(self.data['R'], self.data['z'])

        print('Data loaded')

        self.extrapolate_entropy(10 * Rearth, 2 * Rearth)

        plt.imshow(self.data['s'].value, cmap='plasma')
        plt.show()

    # converts r and theta values into indexes compatible with the data array
    def index_polar(self, r, theta):
        R, z = r * np.sin(theta), r * np.cos(theta)
        i_R = np.int32(R / self.cell_size)
        i_z = np.int32((z + self.size) / self.cell_size)
        return i_z, i_R

    # extrapolates the entropy outwards
    def extrapolate_entropy(self, R_min, z_min):
        print('Extrapolating entropy...')

        # mask of all points outside the high definition region
        extrapolation_mask = ((self.data['R'] / R_min) ** 2 + (self.data['z'] / z_min) ** 2 > 1)
        plot_mask = ((self.data['R'] / R_min) ** 2 + (self.data['z'] / z_min) ** 2 > 0.5)

        # gets the r and theta values along the extrapolation boundary
        n_theta = 100
        A1_theta = np.arange(0, n_theta + 1) * ((np.pi) / n_theta)
        A1_x = (R_min * 0.98 * np.sin(A1_theta)) / self.cell_size
        A1_y = (z_min * 0.98 * np.cos(A1_theta)) / self.cell_size - self.resolution / 2

        # gets the entropy on the extrapolation boundary
        indexes = (np.int32(A1_y), np.int32(A1_x))
        A1_s = self.data['s'][tuple(indexes)]
        self.data['s'][tuple(indexes)] = np.NaN

        # creates a function that gets the entropy on the extrapolation boundary as a function of theta
        s_interp = interp1d(A1_theta, A1_s)
        s_extrapolation = s_interp(self.data['theta'])



        # extrapolates the entropy outwards
        self.data['s'] = np.where(extrapolation_mask, s_extrapolation, self.data['s']) * J / K / kg
        self.data['s'] = np.where(plot_mask, self.data['s'], np.NaN) * J / K / kg

        plt.imshow(self.data['s'].value, cmap='gist_rainbow')
        plt.show()
        plt.plot(A1_theta, A1_s)
        plt.show()


snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
photosphere(snapshot1, 10 * Rearth, 500)
