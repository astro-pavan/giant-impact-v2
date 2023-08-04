# extracts data from a snapshot and analyses it, producing a 2D photosphere model

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from scipy.interpolate import interp1d
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import cm, g, Rearth, J, K, kg, G
from tqdm import tqdm

from snapshot_analysis import snapshot, gas_slice
from forsterite import rho, T, alpha

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
            angular_momentum_slice = get_slice('specific_angular_momentum')

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
        self.data = get_averaged_data(10)

        # removes half the data array
        for k in self.data.keys():
            self.data[k] = self.data[k][:,int(resolution/2):]

        # extends the data array
        extend_R = int(resolution)
        extend_z = int(resolution / 2)
        for k in self.data.keys():
            unit = self.data[k].units
            self.data[k] = np.pad(self.data[k], ((extend_z, extend_z), (0, extend_R)), 'constant') * unit
            #self.data[k] = np.pad(self.data[k], ((extend_z, extend_z), (0, 0)), 'constant') * unit

        # calculating coordinates
        rows, cols = self.data['rho'].shape[0], self.data['rho'].shape[1]
        self.z_height = int(rows/2)
        iX, iY = np.meshgrid(np.arange(cols), np.arange(rows))
        self.data['R'] = ((iX + 0.5) / pixels_per_Rearth) * Rearth
        self.data['z'] = ((iY - self.z_height + 0.5) / pixels_per_Rearth) * Rearth
        self.data['r'] = np.sqrt(self.data['R'] ** 2 + self.data['z'] ** 2)
        self.data['theta'] = np.arctan2(self.data['R'], self.data['z'])

        print('Data loaded')

        self.extrapolate_entropy(self.snapshot.HD_limit_R * 0.8, self.snapshot.HD_limit_z * 0.8)

        self.extrapolate_pressure(self.snapshot.HD_limit_R, self.snapshot.HD_limit_z)

        self.data['alpha'] = alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        self.data['alpha_v'] = alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)

        self.plot('P')
        self.plot('s')
        self.plot('rho')
        self.plot('T')
        self.plot('alpha')
        self.plot('alpha_v')


    def plot(self, parameter):
        central_mask = np.sqrt(self.data['R'] ** 2 + self.data['z'] ** 2) > 3*Rearth
        z = self.data[parameter].value if parameter == 'T' or parameter == 's' else np.log10(self.data[parameter].value)
        z = np.where(central_mask, z, np.NaN)
        plt.imshow(z, cmap='jet')
        plt.show()

    # converts r and theta values into indexes compatible with the data array
    def index_polar(self, r, theta):
        R, z = r * np.sin(theta), r * np.cos(theta)
        i_R = np.int32(R / self.cell_size)
        i_z = np.int32((z + self.size) / self.cell_size)
        return i_z, i_R

    # converts R and z values into indexes compatible with the data array
    def index_cylinder(self, R, z):
        i_R = np.int32(R / self.cell_size)
        i_z = np.int32((z / self.cell_size) + self.z_height)
        return i_z, i_R

    # extrapolates entropy adiabatically with elliptic coords
    def extrapolate_entropy(self, R_min, z_min):

        n_v = 400
        v = np.arange(n_v + 1) * (np.pi/n_v) - np.pi / 2
        a = np.sqrt(R_min ** 2 - z_min ** 2)
        u = np.arccosh(R_min/a)
        R = a * np.cosh(u) * np.cos(v) * 0.98
        z = a * np.sinh(u) * np.sin(v) * 0.98

        indexes = self.index_cylinder(R, z)
        s = self.data['s'][tuple(indexes)]
        s_at_boundary = interp1d(v, s)

        x, y = self.data['R'], self.data['z']
        A2_v = np.sign(y) * np.arccos((np.sqrt((x+a)**2 + y**2) - np.sqrt((x-a)**2 + y**2)) / (2*a))

        s_extrapolation = s_at_boundary(A2_v)
        extrapolation_mask = ((self.data['R'] / R_min) ** 2 + (self.data['z'] / z_min) ** 2 > 1)

        self.data['s'] = np.where(extrapolation_mask, s_extrapolation, self.data['s']) * J / K / kg

    def dPdR(self, P, S, R, z):
        omega = self.snapshot.best_fit_rotation_curve(R)
        grav = (G * self.snapshot.total_mass) / ((R ** 2 + z ** 2) ** (3/2))
        density = rho(S, P)
        result = density * R * (omega ** 2 - grav)
        return result

    def dPdz(self, P, S, R, z):
        grav = (G * self.snapshot.total_mass) / ((R ** 2 + z ** 2) ** (3 / 2))
        return rho(S, P) * z * (- grav)

    def extrapolate_pressure(self, R_min, z_min):

        i_R_limit = self.index_cylinder(R_min, 0)[1]
        i_R_end = self.data['R'].shape[1]
        i_z_min = self.index_cylinder(0, -z_min)[0]
        i_z_max = self.index_cylinder(0, +z_min)[0]
        i_z_end = self.data['R'].shape[0]

        self.data['dR'] = np.roll(self.data['R'], -1, axis=1) - self.data['R']
        self.data['dz'] = np.roll(self.data['z'], -1, axis=0) - self.data['z']

        self.data['dP_R'] = np.zeros_like(self.data['R'])
        self.data['dP_z'] = np.zeros_like(self.data['R'])

        def propagate_R(j, i_min, i_max):
            P, S = self.data['P'][i_min:i_max, j], self.data['s'][i_min:i_max, j]
            R, z = self.data['R'][i_min:i_max, j], self.data['z'][i_min:i_max, j]
            dR = self.data['dR'][i_min:i_max, j]
            dPdR = self.dPdR(P, S, R, z)
            dP_R = dPdR * dR

            self.data['P'][i_min:i_max, j + 1] = self.data['P'][i_min:i_max, j] + dP_R

        def propagate_z(i, j_min, j_max, reverse):
            P, S = self.data['P'][i, j_min:j_max], self.data['s'][i, j_min:j_max]
            R, z = self.data['R'][i, j_min:j_max], self.data['z'][i, j_min:j_max]
            dz = self.data['dz'][i, j_min:j_max]
            dPdz = self.dPdz(P, S, R, z)
            dP_z = dPdz * dz * (-1 if reverse else +1)

            self.data['P'][i + (-1 if reverse else +1), j_min:j_max] = self.data['P'][i, j_min:j_max] + dP_z

        print('Extrapolating pressure...')

        # extrapolate radially at |z| < z_min
        print('Stage 1:')
        for j in tqdm(range(i_R_limit, i_R_end-1)):
            propagate_R(j, i_z_min, i_z_max)

        # extrapolate vertically at R < R_min
        print('Stage 2:')
        for i in tqdm(range(i_z_max, i_z_end-1)):
            propagate_z(i, 0, i_R_limit, False)
        print('Stage 3:')
        for i in tqdm(reversed(range(1, i_z_min))):
            propagate_z(i, 0, i_R_limit, True)

        # extrapolate radially at |z| > z_min
        print('Stage 4:')
        for j in tqdm(range(i_R_limit-1, i_R_end-1)):
            propagate_R(j, 0, i_z_min)
            propagate_R(j, i_z_max, i_z_end)

        print('Pressure extrapolated')

        print('Calculating density and temperature:')
        for i in tqdm(range(0, i_z_end)):
            self.data['rho'][i, :] = rho(self.data['s'][i, :], self.data['P'][i, :])
            self.data['T'][i, :] = T(self.data['s'][i, :], self.data['P'][i, :])


snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
# slice1 = gas_slice(snapshot1, size=12, rotate_vector=(1, 0, 0))
# slice1.full_plot()
photosphere(snapshot1, 12 * Rearth, 500)
