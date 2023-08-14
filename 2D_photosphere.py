# extracts data from a snapshot and analyses it, producing a 2D photosphere model

import matplotlib.pyplot as plt
import numpy as np
import unyt
from matplotlib import ticker
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import cm, g, Rearth, J, K, kg, G, m, stefan_boltzmann_constant, W, dimensionless, Pa, s
from tqdm import tqdm

from snapshot_analysis import snapshot, gas_slice, data_labels
from forsterite import rho_EOS, T_EOS, alpha, cs_EOS, u_EOS
import forsterite


def get_v(R, z, a):
    v = np.sign(z) * np.arccos((np.sqrt((R + a) ** 2 + z ** 2) - np.sqrt((R - a) ** 2 + z ** 2)) / (2 * a))
    return np.nan_to_num(v)


class photosphere_cyl:

    def __init__(self, snapshot, size, resolution):

        self.data, self.data_sphere = {}, {}
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
        self.R_size, self.z_size = cols * self.cell_size, self.z_height * self.cell_size
        iX, iY = np.meshgrid(np.arange(cols), np.arange(rows))
        self.data['R'] = ((iX + 0.5) / pixels_per_Rearth) * Rearth
        self.data['z'] = ((iY - self.z_height + 0.5) / pixels_per_Rearth) * Rearth
        self.data['r'] = np.sqrt(self.data['R'] ** 2 + self.data['z'] ** 2)
        self.data['theta'] = np.arctan2(self.data['R'], self.data['z'])

        self.data['c_s'] = np.zeros_like(self.data['theta']) * (m / unyt.s)
        self.data['u'] = np.zeros_like(self.data['theta']) * (J / kg)

        print('Data loaded')

        self.extrapolate_entropy(self.snapshot.HD_limit_R * 0.85, self.snapshot.HD_limit_z * 0.85)

        self.extrapolate_pressure(self.snapshot.HD_limit_R * 0.85, self.snapshot.HD_limit_z * 0.85)

        self.data['alpha'] = alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        self.data['alpha_v'] = alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)

        self.data['u'] = u_EOS(self.data['rho'], self.data['T']).T

        self.data['t_sound'] = self.data['r'] / self.data['c_s']
        self.data['t_sound'].convert_to_units(unyt.s)

        self.data['dV'] = 2*np.pi*self.data['R']*self.data['dz']*self.data['dR']
        self.data['m'] = self.data['dV'] * self.data['rho']
        self.data['m'].convert_to_units(kg)

        self.data['L'] = 2*np.pi*self.data['R']*self.data['dz']*stefan_boltzmann_constant*(self.data['T'] ** 4)*\
                         self.data['alpha']*self.data['dR']
        self.data['dL'] = np.abs(np.roll(self.data['L'], -1, axis=1) - self.data['L'])

        self.data['E'] = self.data['dV']*self.data['rho']*self.data['u']
        self.data['E'].convert_to_units(J)
        self.data['L'].convert_to_units(W)
        self.data['t_cool'] = self.data['E'] / self.data['dL']
        self.data['t_cool'].convert_to_units(unyt.s)

        self.plot('m')
        self.plot('E')
        self.plot('L')
        self.plot('dL')
        self.plot('t_cool')
        self.plot('t_sound')

        self.plot('P')
        self.plot('s')
        # self.plot('rho')
        # self.plot('T')
        self.plot('alpha')
        #self.plot('alpha_v')

        colours = ['red', 'orange', 'gold', 'green', 'blue', 'purple']
        z = [0, 0.2, 0.4, 0.6, 0.8, 1]
        theta = [np.pi * 0.5, np.pi * 0.4, np.pi * 0.3, np.pi * 0.2, np.pi * 0.1, 0]

        # for i in range(6):
        #     i_z, i_R = self.index_cylinder(1*Rearth, z[i]*Rearth)
        #     plt.plot(self.data['s'][i_z, i_R:].value, self.data['P'][i_z, i_R:].value, label=f'z={z[i]}', c=colours[i])

        for i in range(6):
            indexes = self.index_polar(np.linspace(1, 15) * Rearth, theta[i])
            s = self.data['s'][tuple(indexes)].value
            p = self.data['P'][tuple(indexes)].value
            plt.plot(s, p, label=f'theta={theta[i]}', c=colours[i])

        plt.plot(forsterite.NewEOS.vc.Sl, forsterite.NewEOS.vc.Pl, 'k--')
        plt.plot(forsterite.NewEOS.vc.Sv, forsterite.NewEOS.vc.Pv, 'k--')
        plt.xlim(5000, 11000)
        plt.ylim(1e2, 1e10)
        plt.xlabel(data_labels['s'])
        plt.ylabel(data_labels['P'])
        plt.yscale('log')
        plt.legend()
        plt.show()

    def plot(self, parameter):
        central_mask = np.sqrt(self.data['R'] ** 2 + self.data['z'] ** 2) > 3*Rearth
        z = self.data[parameter].value if parameter == 'T' or parameter == 's' else np.log10(self.data[parameter].value)
        #z = np.where(central_mask, z, np.NaN)
        img = plt.imshow(z, cmap='jet')
        plt.colorbar(img, label=f'log[{data_labels[parameter]}]')
        plt.show()

    def sphere_plot(self, parameter):
        z = np.log10(self.data_sphere[parameter].value)
        x, y = self.data_sphere['R'].value, self.data_sphere['z'].value

        plt.contourf(x, y, z, 100, cmap='jet')
        plt.show()

    # converts r and theta values into indexes compatible with the data array
    def index_polar(self, r, theta):
        R, z = r * np.sin(theta), r * np.cos(theta)
        i_R = np.int32(R / self.cell_size)
        i_z = np.int32((z / self.cell_size) + self.z_height)
        return i_z, i_R

    # converts R and z values into indexes compatible with the data array
    def index_cylinder(self, R, z):
        i_R = np.int32(R / self.cell_size)
        i_z = np.int32((z / self.cell_size) + self.z_height)
        return i_z, i_R

    # extrapolates entropy adiabatically with elliptic coords
    def extrapolate_entropy(self, R_min, z_min):

        print(f'Extrapolating from R = {R_min}, z = {z_min}')

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

            omega = self.snapshot.best_fit_rotation_curve(R)
            grav = (G * self.snapshot.total_mass) / ((R ** 2 + z ** 2) ** (3 / 2))
            density, temp = rho_EOS(S, P), T_EOS(S, P)

            self.data['rho'][i_min:i_max, j] = density
            self.data['T'][i_min:i_max, j] = temp
            self.data['c_s'][i_min:i_max, j] = cs_EOS(density, temp)
            #self.data['u'][i_min:i_max, j] = u_EOS(density, temp)

            dR = self.data['dR'][i_min:i_max, j]
            dPdR = density * R * (omega ** 2 - grav)
            dP_R = dPdR * dR
            new_P = self.data['P'][i_min:i_max, j] + dP_R

            assert(np.any(new_P > 0))

            self.data['P'][i_min:i_max, j + 1] = new_P

        def propagate_z(i, j_min, j_max, reverse):
            P, S = self.data['P'][i, j_min:j_max], self.data['s'][i, j_min:j_max]
            R, z = self.data['R'][i, j_min:j_max], self.data['z'][i, j_min:j_max]

            grav = (G * self.snapshot.total_mass) / ((R ** 2 + z ** 2) ** (3 / 2))
            density, temp = rho_EOS(S, P), T_EOS(S, P)

            self.data['rho'][i, j_min:j_max] = density
            self.data['T'][i, j_min:j_max] = temp
            self.data['c_s'][i, j_min:j_max] = cs_EOS(density, temp)
            #self.data['u'][i, j_min:j_max] = u_EOS(density, temp)

            dz = self.data['dz'][i, j_min:j_max]
            dPdz = density * z * (- grav)
            dP_z = dPdz * dz * (-1 if reverse else +1)

            self.data['P'][i + (-1 if reverse else +1), j_min:j_max] = self.data['P'][i, j_min:j_max] + dP_z

        print('Extrapolating pressure...')

        # extrapolate radially at |z| < z_min
        print('Stage 1:')
        for j in tqdm(range(i_R_limit-1, i_R_end-1)):
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

    def spherical_transform(self):

        n_r, n_theta = 600, 400
        r = np.linspace(0.1, self.z_size.value * 0.9, num=n_r) * Rearth
        theta = np.arange(n_theta + 1) * (np.pi/n_theta)

        rs, thetas = np.meshgrid(r, theta)
        indexes = self.index_polar(rs, thetas)

        for k in self.data.keys():
            self.data_sphere[k] = self.data[k][tuple(indexes)]

        self.data_sphere['r'] = rs * Rearth
        self.data_sphere['theta'] = thetas

    def spherical_extrapolation(self):

        def propagate_r(j):

            P, S = self.data['P'][:, j], self.data['s'][:, j]
            R, z = self.data['R'][:, j], self.data['z'][:, j]

            omega = self.snapshot.best_fit_rotation_curve(R)
            grav = (G * self.snapshot.total_mass) / ((R ** 2 + z ** 2) ** (3 / 2))
            density, temp = rho_EOS(S, P), T_EOS(S, P)

            self.data['rho'][:, j] = density
            self.data['T'][:, j] = temp
            #self.data['u'][i_min:i_max, j] = u_interpolation(density, temp)

            dR = self.data['dR'][:, j]
            dPdR = density * R * (omega ** 2 - grav)
            dP_R = dPdR * dR
            new_P = self.data['P'][:, j] + dP_R

            assert(np.any(new_P > 0))

            self.data['P'][: j + 1] = new_P

units = {
    'r': m,
    'R': m,
    'z': m,
    'theta': dimensionless,
    'rho': kg * (m ** -3),
    'T': K,
    'P': Pa,
    'S': J / K / kg,
    'u': J / kg,
    'm': kg,
    'V': m ** 3,
    'E': J,
    'h': (m ** 2) / s
}


class photosphere_sph:

    def __init__(self, snapshot, sample_size, max_size, resolution):

        sample_size.convert_to_units(Rearth)
        self.snapshot = snapshot

        self.data = {}
        n_theta, n_phi = 400, 10

        r, theta = np.meshgrid(np.linspace(0, sample_size.value * 0.95, num=int(resolution / 2)),
                               np.arange(n_theta + 1) * (np.pi / n_theta))
        pixel_size = sample_size.value / (resolution / 2)

        i_R = np.int32((r * np.sin(theta) / pixel_size) + (resolution / 2))
        i_z = np.int32((r * np.cos(theta) / pixel_size) + (resolution / 2))
        indexes = i_z, i_R

        def get_section(phi):
            data = {}

            # properties used to load the slices
            center = snapshot.center_of_mass
            rotate_z = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='z')
            rotate_x = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='x')
            matrix = np.matmul(rotate_x, rotate_z)
            limits = [center[0] - sample_size, center[0] + sample_size, center[1] - sample_size, center[1] + sample_size]

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

                property_slice = mass_weighted_slice / mass_slice

                return property_slice[tuple(indexes)]

            # loading slices of each property
            data['T'] = get_slice('temperatures')
            data['P'], data['s'] = get_slice('pressures'), get_slice('entropy')
            data['h'] = get_slice('specific_angular_momentum')

            mass_slice.convert_to_units(g / cm ** 3)

            data['rho'] = mass_slice[tuple(indexes)]

            data['r'], data['theta'] = r * Rearth, theta

            return data

        print('Loading data into photosphere model:')

        # loads multiple sections at different phi angles and averages them
        self.data = get_section(0)

        for i in tqdm(range(1, n_phi)):
            vals = get_section(np.pi / n_phi * i)
            for k in self.data.keys():
                self.data[k] = (i * self.data[k] + vals[k]) / (i + 1)

        extend_r = int((max_size.value - sample_size.value) / pixel_size)

        for k in self.data.keys():
            unit = 1 if k == 'theta' else self.data[k].units
            if k == 'r':
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'linear_ramp', end_values=(0, max_size.value)) * unit
            elif k == 'theta':
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'edge') * unit
            else:
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'constant') * unit

        self.data['R'] = self.data['r'] * np.sin(self.data['theta'])
        self.data['z'] = self.data['r'] * np.cos(self.data['theta'])

        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[0] / max_size
        self.R_min, self.z_min = snapshot.HD_limit_R * 0.95, snapshot.HD_limit_z * 0.95
        self.linear_eccentricity = np.sqrt(self.R_min ** 2 - self.z_min ** 2)
        self.central_mass = self.snapshot.total_mass

        self.entropy_extrapolation = self.extrapolate_entropy()

    def plot(self, parameter, log):
        z = self.data[parameter].value
        plt.contourf(self.data['R'].value, self.data['z'].value, np.log10(z) if log else z, 80, cmap='jet')
        plt.colorbar(label=f'log[{data_labels[parameter]}]')
        plt.show()

    def get_index(self, r, theta):
        i_r = np.int32(r * self.i_per_r)
        i_theta = np.int32(theta * self.i_per_theta)
        return i_theta, i_r

    def extrapolate_entropy(self):

        print(f'Extrapolating from R = {self.R_min}, z = {self.z_min}')

        n_v = 400
        v = np.arange(n_v + 1) * (np.pi / n_v) - np.pi / 2
        a = self.linear_eccentricity
        u = np.arccosh(self.R_min / a)
        z, R = a * np.cosh(u) * np.cos(v) * 0.98, a * np.sinh(u) * np.sin(v) * 0.98

        r, theta = np.sqrt(R ** 2 + z ** 2), np.arctan2(z, R)

        indexes = self.get_index(r, theta)
        s = self.data['s'][tuple(indexes)]
        entropy_extrapolation = interp1d(v, s, bounds_error=False, fill_value='extrapolate')

        x, y = self.data['R'], self.data['z']
        A2_v = get_v(x, y, a)

        extrapolation_mask = ((self.data['R'] / self.R_min) ** 2 + (self.data['z'] / self.z_min) ** 2 > 1)
        self.data['s'] = np.where(extrapolation_mask, entropy_extrapolation(A2_v), self.data['s']) * J / K / kg

        return entropy_extrapolation

    def dPdr(self, P, r, theta):
        gravity = - (G * self.snapshot.total_mass) / (r ** 2)
        R = r * np.sin(theta)
        centrifugal = R * (self.snapshot.best_fit_rotation_curve(R) ** 2) * np.sin(theta)
        v = get_v(R, r * np.cos(theta), self.linear_eccentricity)
        S = self.entropy_extrapolation(v)
        rho = rho_EOS(S, P)
        return rho * (gravity + centrifugal)

    def extrapolate_pressure_v1(self):

        self.data['dr'] = np.roll(self.data['r'], -1, axis=1) - self.data['r']

        def propagate_r(j):

            P, S = self.data['P'][:, j], self.data['s'][:, j]
            r, R = self.data['r'][:, j], self.data['R'][:, j]

            omega = self.snapshot.best_fit_rotation_curve(R)
            gravity = - (G * self.snapshot.total_mass) / (r ** (3 / 2))
            centrifugal = R * (omega ** 2)
            density, temp = rho_EOS(S, P), T_EOS(S, P)

            self.data['rho'][:, j] = density
            self.data['T'][:, j] = temp
            # self.data['u'][i_min:i_max, j] = u_interpolation(density, temp)

            dr = self.data['dr'][:, j]
            dPdr = density * (gravity + centrifugal)
            dP = dPdr * dr
            new_P = self.data['P'][:, j] + dP

            assert (np.any(new_P > 0))

            self.data['P'][: j + 1] = new_P

    def extrapolate_pressure_v2(self):

        for i in tqdm(range(self.data['r'].shape[0])):

            theta = self.data['theta'][i, 0]
            r_0 = np.sqrt((self.R_min*np.sin(theta)) ** 2 + (self.z_min*np.cos(theta)) ** 2)
            j_0 = self.get_index(r_0, theta)[1]
            P_0 = self.data['P'][i, j_0]

            f = lambda P, r: self.dPdr(P, r, theta)

            r_solution = self.data['r'][i, j_0:]
            P_solution = odeint(f, P_0, r_solution)

            self.data['P'][i, j_0] = P_solution


snapshot1 = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
#p1 = photosphere_cyl(snapshot1, 15 * Rearth, 200)
p2 = photosphere_sph(snapshot1, 12 * Rearth, 30 * Rearth, 500)
p2.extrapolate_pressure_v2()
