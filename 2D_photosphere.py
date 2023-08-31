# extracts data from a snapshot and analyses it, producing a 2D photosphere model

import matplotlib.pyplot as plt
import numpy as np
import unyt
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import cm, g, Rearth, J, K, kg, G, m, stefan_boltzmann_constant, W, dimensionless, Pa, s
from tqdm import tqdm
from multiprocessing import Pool
import sys, uuid

from snapshot_analysis import snapshot, gas_slice, data_labels
import forsterite2 as fst

sigma = 5.670374419e-8
L_sun = 3.828e26
pi = np.pi


def cos(theta):
    return np.cos(theta)


def sin(theta):
    return np.sin(theta)


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def get_v(R, z, a):
    v = np.sign(z) * np.arccos((np.sqrt((R + a) ** 2 + z ** 2) - np.sqrt((R - a) ** 2 + z ** 2)) / (2 * a))
    return np.nan_to_num(v)


def shift_right(arr):
    result = np.roll(arr, +1, axis=1)
    result[:, 0] = np.zeros_like(result[:, 0])
    return result


def shift_up(arr):
    result = np.roll(arr, -1, axis=0)
    result[-1, :] = np.zeros_like(result[-1, :])
    return result


def shift_down(arr):
    result = np.roll(arr, +1, axis=0)
    result[0, :] = np.zeros_like(result[0, :])
    return result


class photosphere_sph:

    # sample size and max size both have units
    def __init__(self, snapshot, sample_size, max_size, resolution, n_theta=100, n_phi=10):

        self.luminosity = None
        self.phot_indexes = None
        self.j_phot = None
        self.L_phot = None
        self.r_phot = None
        self.R_phot, self.z_phot = None, None

        sample_size.convert_to_units(Rearth)
        self.snapshot = snapshot
        self.data = {}

        # calculates the indexes to sample from to fill the array
        r, theta = np.meshgrid(np.linspace(0, sample_size.value * 0.95, num=int(resolution / 2)) * Rearth,
                               np.arange(n_theta + 1) * (np.pi / n_theta))
        pixel_size = sample_size.value / (resolution / 2)
        i_R = np.int32((r.value * np.sin(theta) / pixel_size) + (resolution / 2))
        i_z = np.int32((r.value * np.cos(theta) / pixel_size) + (resolution / 2))
        indexes = i_z, i_R
        extend_r = int((max_size.value - sample_size.value) / pixel_size)

        # loads a cross-section of the simulation at an angle phi
        def get_section(phi):
            data = {}

            # properties used to load the slices
            center = snapshot.center_of_mass
            rotate_z = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='z')
            rotate_x = rotation_matrix_from_vector([np.cos(phi), np.sin(phi), 0], axis='x')
            matrix = np.matmul(rotate_x, rotate_z)
            limits = [center[0] - sample_size, center[0] + sample_size, center[1] - sample_size,
                      center[1] + sample_size]

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
            temperatures = get_slice('temperatures')
            pressures, entropies = get_slice('pressures'), get_slice('entropy')
            angular_momenta = get_slice('specific_angular_momentum')

            # convert data to MKS
            mass_slice.convert_to_mks()
            temperatures.convert_to_mks()
            pressures.convert_to_mks()
            entropies.convert_to_mks()
            angular_momenta.convert_to_mks()
            r.convert_to_mks()

            # put data in dictionary
            data['r'], data['theta'] = r, theta
            data['rho'], data['T'] = mass_slice[tuple(indexes)].value, temperatures
            data['P'], data['s'] = pressures, entropies
            data['h'] = angular_momenta

            return data

        print('Loading data into photosphere model:')

        # loads multiple sections at different phi angles and averages them
        self.data = get_section(0)

        for i in tqdm(range(1, n_phi)):
            vals = get_section(np.pi / n_phi * i)
            for k in self.data.keys():
                self.data[k] = (i * self.data[k] + vals[k]) / (i + 1)

        max_size.convert_to_mks()

        # extends the data arrays ready for extrapolation
        for k in self.data.keys():
            if k == 'r':
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'linear_ramp',
                                      end_values=(0, max_size.value))
            elif k == 'theta':
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'edge')
            else:
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'constant')

        # calculates the R and z coordinates for each point
        self.data['R'] = self.data['r'] * np.sin(self.data['theta'])
        self.data['z'] = self.data['r'] * np.cos(self.data['theta'])

        self.data['dr'] = np.roll(self.data['r'], -1, axis=1) - self.data['r']
        self.data['dr'][:, -1] = self.data['dr'][:, -2]
        self.data['d_theta'] = np.full_like(self.data['dr'], np.pi / n_theta)

        data = self.data
        r, dr = data['r'], data['dr']
        theta, d_theta = data['theta'], data['d_theta']

        data['A_r-'] = 2 * pi * (r ** 2) * (cos(theta) - cos(theta + d_theta))
        data['A_r+'] = 2 * pi * ((r + dr) ** 2) * (cos(theta) - cos(theta + d_theta))

        for j in range(data['r'].shape[1]):
            r_test = r[0, j]
            assert (np.sum(data['A_r-'][:, j]) - 4 * pi * (r_test ** 2)) / 4 * pi * (r_test ** 2) < 0.01

        data['A_theta-'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta)
        data['A_theta+'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta + d_theta)

        data['A_theta+'][-1, :] = np.zeros_like(data['A_theta+'][-1, :])

        data['V'] = pi * ((r + dr) ** 2 - r ** 2) * (cos(theta) - cos(theta + d_theta))

        # these values are used to calculate the index in the array for a given r and theta
        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[1] / max_size.value

        # values used to get the elliptical surface for the start of the extrapolation
        self.R_min, self.z_min = snapshot.HD_limit_R.value * 0.95, snapshot.HD_limit_z.value * 0.95
        self.linear_eccentricity = np.sqrt(
            self.R_min ** 2 - self.z_min ** 2)  # linear eccentricity of the extrapolation surface

        self.central_mass = self.snapshot.total_mass.value

        self.t_dyn = np.sqrt((max_size.value ** 3) / (6.674e-11 * self.central_mass))

        # extrapolation performed here
        self.entropy_extrapolation = self.extrapolate_entropy()
        self.extrapolate_pressure_v2()
        self.calculate_EOS()

        self.data['omega'] = self.data['h'] * (self.data['R'] ** -2)

        self.data['L_r+'] = np.zeros_like(self.data['r'])
        self.data['L_theta-'] = np.zeros_like(self.data['r'])
        self.data['L_theta+'] = np.zeros_like(self.data['r'])

        js = np.zeros(r.shape[0])
        for i in range(r.shape[0]):
            js[i] = np.nanargmax(self.data['rho'][i, :])[()]
        ind = np.arange(0, r.shape[0]), np.int32(js)

        self.surf_indexes = tuple(ind)
        self.j_surf = np.int32(js)

        # self.data['cross_section'] = np.minimum(self.data['alpha_v'] * self.data['V'], self.data['A_r+'])
        # self.data['A_factor'] = self.data['cross_section'] / self.data['A_r+']

        # self.data['puff'] = (((5 * 6371000)/ self.data['r']) ** 2) * ((5000/self.data['T']) ** 4)

        # self.data['t_cool'] = self.data['E'] / (self.data['T'] ** 4 * sigma * self.data['A_r+'])
        #
        # self.data['tau'] = self.data['dr'] * self.data['alpha']
        # self.data['tau_v'] = self.data['dr'] * self.data['alpha_v']
        #
        # self.data['tau_v_2'] = self.data['r'] * self.data['d_theta'] * self.data['alpha_v']
        #
        # plt.imshow(self.data['E'])
        # plt.show()

    def plot(self, parameter, log=True, contours=None, cmap='cubehelix', plot_photosphere=False):
        vals = np.log10(self.data[parameter]) if log else self.data[parameter]
        R, z = self.data['R'] * m, self.data['z'] * m
        R.convert_to_units(Rearth)
        z.convert_to_units(Rearth)

        plt.figure(figsize=(8, 10))

        # min_tick, max_tick = np.floor(np.nanmin(vals)), np.ceil(np.nanmax(vals))
        # if log:
        #     if max_tick - min_tick > 20:
        #         step = 2
        #     else:
        #         step = 1
        #     # ticks = np.arange(min_tick, max_tick, step)
        # else:
        #     tick_range = np.nanmax(vals) - np.nanmin(vals)
        #     tick_interval = 1000
        #     # ticks = np.arange(np.floor(np.nanmin(vals) / tick_interval) * tick_interval,
        #     #                   np.ceil(np.nanmax(vals) / tick_interval) * tick_interval + tick_interval,
        #     #                   tick_interval)

        plt.contourf(R, z, vals, 200, cmap=cmap)
        cbar = plt.colorbar(label=data_labels[parameter] if not log else '$\log_{10}$[' + data_labels[parameter] + ']')
        plt.xlabel(data_labels['R'])
        plt.ylabel(data_labels['z'])
        #plt.ylim([-10, 10])
        
        if plot_photosphere:
            plt.plot(self.R_phot / 6371000, self.z_phot / 6371000, 'w--')

        # theta = np.linspace(0, np.pi)
        # plt.plot(1.5 * np.sin(theta), 1.5 * np.cos(theta), 'w--')

        if contours is not None:
            cs = plt.contour(R, z, vals, contours, colors='black', linestyles='dashed')
            plt.clabel(cs, contours, colors='black')

        # vals = np.nan_to_num(vals, posinf=0)
        #
        # if log:
        #     ticks = np.arange(int(np.nanmin(vals)), int(np.nanmax(vals)))
        #     cbar.set_ticks(ticks)
        plt.show()

    def get_index(self, r, theta):
        i_r = np.int32(r * self.i_per_r)
        i_theta = np.int32(theta * self.i_per_theta)
        return i_theta, i_r

    def extrapolate_entropy(self):

        print(f'Extrapolating from R = {self.R_min / 6371000:.2f}, z = {self.z_min / 6371000:.2f}')

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
        self.data['s'] = np.where(extrapolation_mask, entropy_extrapolation(A2_v), self.data['s'])

        def funct(r, theta):
            v = get_v(r * np.sin(theta), r * np.cos(theta), self.linear_eccentricity)
            return entropy_extrapolation(v)

        return funct

    def dPdr(self, P, r, theta, S_funct=None):
        gravity = - (6.674e-11 * self.central_mass) / (r ** 2)

        R = r * np.sin(theta)
        centrifugal = R * (self.snapshot.best_fit_rotation_curve_mks(R) ** 2) * np.sin(theta)

        S = self.entropy_extrapolation(r, theta)

        rho = fst.rho_EOS(S, P)

        result = rho * (gravity + centrifugal)

        # if np.isnan(result).any():
        #     print(f'P = {P} S = {S} rho = {rho}')

        return np.nan_to_num(result)

    def extrapolate_pressure_v2(self):

        @globalize
        def extrapolate(i):

            theta = self.data['theta'][i, 0]
            print(f'Extrapolating at theta = {theta:.3f}...')

            r_0 = np.sqrt((self.R_min * np.sin(theta)) ** 2 + (self.z_min * np.cos(theta)) ** 2)
            j_0 = self.get_index(r_0, theta)[1]
            P_0 = self.data['P'][i, j_0]

            f = lambda P, r: self.dPdr(P, r, theta)

            r_solution = self.data['r'][i, j_0:]
            if P_0 != np.inf:
                P_solution = odeint(f, P_0, r_solution)
            else:
                P_solution = np.full_like(self.data['r'][i, j_0:], 0)

            P_solution = np.nan_to_num(P_solution)

            print(f'Extrapolation at theta = {theta:.3f} complete')
            return P_solution.T, j_0, i

        if __name__ == '__main__':
            pool = Pool(7)
            results = pool.map(extrapolate, range(self.data['r'].shape[0]))

        for r in results:
            i, j_0 = r[2], r[1]
            self.data['P'][i:i + 1, j_0:] = r[0]

        #         self.data['P'][i:i+1, j_0:] = P_solution.T

        self.data['P'] = np.nan_to_num(self.data['P'])

        self.data['rho'] = fst.rho_EOS(self.data['s'], self.data['P'])
        self.data['T'] = fst.T1_EOS(self.data['s'], self.data['P'])

    def calculate_EOS(self):
        self.data['alpha'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        self.data['alpha_v'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)
        self.data['tau'] = self.data['alpha'] * self.data['dr']
        self.data['tau_v'] = self.data['alpha_v'] * self.data['dr']

        self.data['u'] = fst.u_EOS(self.data['rho'], self.data['T'])
        self.data['m'] = self.data['rho'] * self.data['V']
        self.data['E'] = self.data['u'] * self.data['m']
        self.data['rho_E'] = self.data['E'] / self.data['V']

        self.data['phase'] = fst.phase(self.data['s'], self.data['P'])

        emissivity = np.minimum((self.data['alpha_v'] * self.data['V']) / self.data['A_r+'], 1)
        L = sigma * self.data['T'] ** 4 * self.data['A_r+'] * emissivity
        self.data['t_cool'] = self.data['E'] / L

    def remove_droplets(self):
        print('Removing droplets...')
        new_S = fst.S_vapor_curve_v(self.data['P'])
        self.data['s'] = np.where(self.data['phase'] == 2, new_S, self.data['s'])
        self.calculate_EOS()

    def get_photosphere(self):
        print('Finding photosphere...')
        
        @globalize
        def optical_depth_integration(i):
            optical_depth = 0
            j = self.data['r'].shape[1] - 1

            while optical_depth < 1:
                tau = self.data['tau_v'][i, j]
                optical_depth += tau
                j -= 1

            T = self.data['T'][i, j]
            r = self.data['r'][i, j]
            L = sigma * T ** 4 * self.data['A_r+'][i, j]
            
            return np.int32(i), np.int32(j), T, r, L
        
        if __name__ == '__main__':
            pool = Pool(7)
            results = pool.map(optical_depth_integration, range(self.data['r'].shape[0]))

        r_phot = np.zeros(self.data['r'].shape[0])
        T_phot, L_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)
        i_phot, j_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)
        R_phot, z_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)

        for res in results:
            i, j, T, r, L = res
            i_phot[i], j_phot[i] = i, j
            r_phot[i] = r
            T_phot[i], L_phot[i] = T, L
            R_phot[i], z_phot[i] = self.data['R'][i, j], self.data['z'][i, j]
        
        self.phot_indexes = tuple((i, j))
        self.luminosity = np.sum(L)
        self.R_phot, self.z_phot = R_phot, z_phot
        print(f'Photosphere found with luminosity = {self.luminosity/3.8e26:.2e} L_sun')

    def get_surface(self):
        pass

    def initial_cool_v2(self, tau_threshold=1e-1, max_time=1e-1):
        print(f'Cooling vapor for {max_time:.2e} s')
        rho = self.data['rho']
        T1 = self.data['T']
        u1 = self.data['u']
        dr = self.data['dr']
        m = self.data['m']
        V = self.data['V']
        A = self.data['A_r+']
        alpha = self.data['alpha_v']

        alpha_threshold = tau_threshold / dr

        T2 = fst.T_alpha_v(rho, alpha_threshold)
        u2 = fst.u_EOS(rho, T2)
        du = u1 - u2
        dE = du / m

        emissivity = np.minimum((alpha * V) / A, 1)
        L = sigma * T1 ** 4 * A * emissivity
        t_cool = dE / L
        cool_check = (alpha > alpha_threshold) & (t_cool < max_time) & (du > 0)
        T2 = np.where(cool_check, T2, T1)
        t_cool = np.where(cool_check, t_cool, 0)
        assert np.all(np.sum(t_cool, axis=1) < max_time)

        self.data['change'] = np.where(cool_check, 1, 0)
        self.data['T'] = T2
        self.data['P'] = fst.P_EOS(rho, T2)
        self.data['s'] = fst.S_EOS(rho, T2)
        self.calculate_EOS()


snapshot1 = snapshot('/home/pavan/PycharmProjects/giant-impact-v2/snapshots/basic_twin/snapshot_0411.hdf5')
# snapshot2 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M1.0_ratio1.00_v1.10_b0.50_spin0.0/output/snapshot_0240.hdf5')
# snapshot3 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.30_spin0.0/output/snapshot_0240.hdf5')
# *snapshot4 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M0.5_ratio0.50_v1.10_b0.50_spin0.0/output/snapshot_0240.hdf5')
# *snapshot5 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.30_spin0.0/output/snapshot_0240.hdf5')
# snapshot6 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M0.1_ratio1.00_v1.10_b0.50_spin0.0/output/snapshot_0240.hdf5')
# nsnapshot7 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M0.5_ratio1.00_v1.10_b0.10_spin0.0/output/snapshot_0240.hdf5')
# snapshot8 = snapshot('/home/pavan/Project/Final_Sims/impact_p1.0e+05_M2.0_ratio1.00_v1.10_b0.50_spin0.0/output/snapshot_0240.hdf5')
p2 = photosphere_sph(snapshot1, 12 * Rearth, 25 * Rearth, 200, n_theta=48)
p2.remove_droplets()
p2.initial_cool_v2(max_time=1, tau_threshold=1e-2)
p2.remove_droplets()
p2.get_photosphere()
p2.plot('t_cool', plot_photosphere=True)
