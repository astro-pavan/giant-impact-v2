# extracts data from a snapshot and analyses it, producing a 2D photosphere model
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import odeint
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import Rearth, m
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import uuid

from snapshot_analysis import snapshot, data_labels
import forsterite2 as fst

cpus = cpu_count()

sigma = 5.670374419e-8
L_sun = 3.828e26
yr = 3.15e7
pi = np.pi
cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def get_v(R, z, a):
    v = np.sign(z) * np.arccos((np.sqrt((R + a) ** 2 + z ** 2) - np.sqrt((R - a) ** 2 + z ** 2)) / (2 * a))
    return np.nan_to_num(v)


class photosphere:

    # sample size and max size both have units
    def __init__(self, snapshot, sample_size=12*Rearth, max_size=30*Rearth, resolution=500, n_theta=100, n_phi=10):

        self.j_surf_1, self.j_surf_2, self.j_phot = np.zeros(n_theta + 1), np.zeros(n_theta + 1), np.zeros(n_theta + 1)
        self.L_surf_1, self.L_surf_2, self.L_phot = 0, 0, 0
        self.R_phot, self.z_phot = np.zeros(n_theta + 1), np.zeros(n_theta + 1)
        self.R_surf_1, self.z_surf_1 = np.zeros(n_theta + 1), np.zeros(n_theta + 1)
        self.R_surf_2, self.z_surf_2 = np.zeros(n_theta + 1), np.zeros(n_theta + 1)

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
            else:
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'edge' if k == 'theta' else 'constant')

        self.n_r, self.n_theta = self.data['r'].shape[1], self.data['r'].shape[0]

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

        data['A_theta-'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta)
        data['A_theta+'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta + d_theta)

        data['A_theta+'][-1, :] = np.zeros_like(data['A_theta+'][-1, :])

        data['V'] = (1/3) * pi * ((r + dr) ** 3 - r ** 3) * (cos(theta) - cos(theta + d_theta))

        # these values are used to calculate the index in the array for a given r and theta
        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[1] / max_size.value

        # values used to get the elliptical surface for the start of the extrapolation
        self.R_min, self.z_min = snapshot.HD_limit_R.value * 0.95, snapshot.HD_limit_z.value * 0.95
        # linear eccentricity of the extrapolation surface
        self.linear_eccentricity = np.sqrt(self.R_min ** 2 - self.z_min ** 2)

        self.central_mass = self.snapshot.total_mass.value
        self.data['omega'] = self.data['h'] * (self.data['R'] ** -2)
        self.t_dyn = np.sqrt((max_size.value ** 3) / (6.674e-11 * self.central_mass))

        # extrapolation performed here
        self.entropy_extrapolation = self.extrapolate_entropy()
        self.hydrostatic_equilibrium(initial_extrapolation=True, solve_log=True)
        self.calculate_EOS()

    def plot(self, parameter, log=True, contours=None, cmap='cubehelix', plot_photosphere=False, limits=None):
        vals = np.log10(self.data[parameter]) if log else self.data[parameter]
        R, z = self.data['R'] * m, self.data['z'] * m
        R.convert_to_units(Rearth)
        z.convert_to_units(Rearth)

        plt.figure(figsize=(10, 8))
        plt.contourf(R, z, vals, 200, cmap=cmap)

        cbar = plt.colorbar(label=data_labels[parameter] if not log else '$\log_{10}$[' + data_labels[parameter] + ']')
        plt.xlabel(data_labels['R'])
        plt.ylabel(data_labels['z'])
        if limits is not None:
            plt.ylim(limits)
        
        if plot_photosphere:
            plt.plot(self.R_phot / 6371000, self.z_phot / 6371000, 'w--')
        plt.plot(self.R_surf_1 / 6371000, self.z_surf_1 / 6371000, 'r--')
        plt.plot(self.R_surf_2 / 6371000, self.z_surf_2 / 6371000, 'b--')
        theta = np.linspace(0, np.pi)
        plt.plot((self.R_min / 6371000) * np.sin(theta), (self.z_min / 6371000) * np.cos(theta), 'k--')

        if log:
            tick_positions = np.arange(np.ceil(np.nanmin(vals)), np.ceil(np.nanmax(vals)))
        else:
            tick_positions = np.arange(np.ceil(np.nanmin(vals / 1000)), np.ceil(np.nanmax(vals / 1000))) * 1000
        cbar.set_ticks(tick_positions)

        if contours is not None:
            cs = plt.contour(R, z, vals, contours, colors='black', linestyles='dashed')
            plt.clabel(cs, contours, colors='black')

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
        omega = self.snapshot.best_fit_rotation_curve_mks(R)
        centrifugal = R * (omega ** 2) * np.sin(theta)

        S = S_funct(r, theta)
        rho = fst.rho_EOS(S, P)

        result = rho * (gravity + centrifugal)
        return np.nan_to_num(result)

    def dlnPdr(self, lnP, r, theta, S_funct=None):
        gravity = - (6.674e-11 * self.central_mass) / (r ** 2)

        R = r * np.sin(theta)
        omega = self.snapshot.best_fit_rotation_curve_mks(R)
        centrifugal = R * (omega ** 2) * np.sin(theta)

        S = S_funct(r, theta)
        rho = fst.rho_EOS(S, np.exp(lnP))

        result = np.exp(-lnP) * rho * (gravity + centrifugal)
        return np.nan_to_num(result)

    def hydrostatic_equilibrium(self, initial_extrapolation=False, solve_log=False):

        print('Solving hydrostatic equilibrium:')

        if initial_extrapolation:
            S_funct = self.entropy_extrapolation

            theta = self.data['theta'][:, 0]
            r_0 = np.sqrt((self.R_min * np.sin(theta)) ** 2 + (self.z_min * np.cos(theta)) ** 2)
            j_start = self.get_index(r_0, theta)[1]

        else:
            r = self.data['r'][0, :]
            theta = self.data['theta'][:, 0]
            S_interp = RegularGridInterpolator((theta, r), np.nan_to_num(self.data['s']), bounds_error=False, fill_value=np.NaN)
            S_funct = lambda x, y: S_interp(fst.make_into_pair_array(y, x))

            r_0 = np.sqrt((2 * np.sin(theta)) ** 2 + (2 * np.cos(theta)) ** 2) * 6371000
            j_start = self.get_index(r_0, theta)[1]

        @globalize
        def extrapolate(i):

            j_0 = np.int32(j_start[i])
            P_0 = self.data['P'][i, j_0] if not solve_log else np.log(self.data['P'][i, j_0])

            if solve_log:
                f = lambda lnP, r: self.dlnPdr(lnP, r, theta[i], S_funct=S_funct)
            else:
                f = lambda P, r: self.dPdr(P, r, theta[i], S_funct=S_funct)

            r_solution = self.data['r'][i, j_0:]
            solution = odeint(f, P_0, r_solution)
            #solution = solve_ivp(lambda t, y: f(y, t), )

            P_solution = np.nan_to_num(np.exp(solution) if solve_log else solution)

            print(u"\u2588", end='')
            return P_solution.T, j_0, i

        pool = Pool(cpus - 1)
        results = pool.map(extrapolate, range(self.n_theta))
        print(' DONE')

        for r in results:
            i, j_0 = r[2], r[1]
            self.data['P'][i:i + 1, j_0:] = r[0]

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
        self.data['vq'] = fst.vapor_quality(self.data['s'], self.data['P'])
        self.data['lvf'] = fst.liquid_volume_fraction(self.data['rho'], self.data['P'], self.data['s'])

        emissivity = np.minimum((self.data['alpha_v'] * self.data['V']) / self.data['A_r+'], 1)
        L = sigma * self.data['T'] ** 4 * self.data['A_r+'] * emissivity
        self.data['t_cool'] = self.data['E'] / L

    def remove_droplets(self):
        condensation_mask = self.data['phase'] == 2
        initial_mass = np.nansum(self.data['m'][condensation_mask])

        print('Removing droplets...')

        new_S = fst.condensation_S(self.data['s'], self.data['P'])
        self.data['s'] = np.where(condensation_mask, new_S, self.data['s'])
        self.data['rho'] = fst.rho_EOS(self.data['s'], self.data['P'])
        self.data['T'] = fst.T1_EOS(self.data['s'], self.data['P'])
        self.calculate_EOS()

        final_mass = np.nansum(self.data['m'][condensation_mask])
        mass_lost = initial_mass - final_mass
        print(f'{mass_lost / 5.972e24:.2e} M_earth lost')

    def get_photosphere(self):
        print('Finding photosphere...')
        
        @globalize
        def optical_depth_integration(i):
            optical_depth = 0
            j = self.n_r - 1

            while optical_depth < 1:
                tau = self.data['tau_v'][i, j]
                optical_depth += tau
                j -= 1

            T = self.data['T'][i, j]
            r = self.data['r'][i, j]
            L = sigma * T ** 4 * self.data['A_r+'][i, j]
            
            return np.int32(i), np.int32(j), T, r, L

        pool = Pool(cpus - 1)
        results = pool.map(optical_depth_integration, range(self.n_theta))

        r_phot = np.zeros(self.n_theta)
        T_phot, L_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)
        i_phot, j_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)
        R_phot, z_phot = np.zeros_like(r_phot), np.zeros_like(r_phot)

        A_total = 0

        for res in results:
            i, j, T, r, L = res
            i_phot[i], j_phot[i] = i, j
            r_phot[i] = r
            T_phot[i], L_phot[i] = T, L
            R_phot[i], z_phot[i] = self.data['R'][i, j], self.data['z'][i, j]
            A_total += self.data['A_r+'][i, j]

        self.L_phot = np.sum(L_phot)
        self.j_phot = j_phot
        self.R_phot, self.z_phot = R_phot, z_phot
        print(f'Photosphere found with luminosity = {self.L_phot / 3.8e26:.2e} L_sun')

    def initial_cool(self, tau_threshold=1e-1, max_time=1e2):
        print(f'Cooling vapor for {max_time:.2e} s')
        rho, T1, u1 = self.data['rho'], self.data['T'], self.data['u']
        dr = self.data['dr']
        m = self.data['m']
        A, V = self.data['A_r+'], self.data['V']
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

        self.data['change'] = np.where(cool_check, 1, 0)
        self.data['T'] = T2
        self.data['P'] = fst.P_EOS(rho, T2)
        self.data['s'] = fst.S_EOS(rho, T2)

        for i in range(self.n_theta):

            self.j_surf_1[i] = np.nanargmax(self.data['rho'][i, :])
            self.R_surf_1[i] = self.data['R'][i, int(self.j_surf_1[i])]
            self.z_surf_1[i] = self.data['z'][i, int(self.j_surf_1[i])]

            self.j_surf_2[i] = np.nanargmax(self.data['change'][i, :])
            self.R_surf_2[i] = self.data['R'][i, int(self.j_surf_2[i])]
            self.z_surf_2[i] = self.data['z'][i, int(self.j_surf_2[i])]

        self.calculate_EOS()

    def set_up_cooling_shells(self):

        self.data['shell'] = np.zeros_like(self.data['r'])

        for i in range(self.n_theta):
            j1 = int(self.j_surf_1[i])
            j2 = int(self.j_surf_2[i])
            j3 = int(self.j_phot[i])
            self.data['shell'][i, j1:j2] = 1
            self.data['shell'][i, j2:j3] = 2
            self.data['shell'][i, j3:] = 3

        inner_mask = self.data['shell'] == 1

        E_outer = np.nansum(self.data['E'][inner_mask])
        m_outer = np.nansum(self.data['m'][inner_mask])
        u_outer = E_outer / m_outer
        self.data['u'][inner_mask] = u_outer
        self.data['T'][inner_mask] = fst.T2_EOS(self.data['u'][inner_mask], self.data['rho'][inner_mask])
        self.data['P'][inner_mask] = fst.P_EOS(self.data['rho'][inner_mask], self.data['T'][inner_mask])
        self.data['s'][inner_mask] = fst.P_EOS(self.data['rho'][inner_mask], self.data['T'][inner_mask])

    def get_shell_area(self, shell_num):
        mask = self.data['shell'] > shell_num
        A = 0
        for i in range(self.n_theta):
            j = np.argmax(mask[i, :])
            A += self.data['A_r+'][i, int(j)]
        return A

    def simple_cooling(self):

        outer_planet_mask = self.data['shell'] == 1
        shell_mask = self.data['shell'] == 2

        E_shell = np.nansum(self.data['E'][shell_mask])
        E_outer_planet = np.sum(self.data['E'][outer_planet_mask])
        E_total = E_shell + E_outer_planet

        A_inner_planet = self.get_shell_area(0)
        T_surf_1 = self.data['T'][0, int(self.j_surf_1[0])]
        L_inner_planet = sigma * T_surf_1 ** 4 * A_inner_planet
        L_total = self.L_phot - L_inner_planet

        t_cool = E_total / L_total

        print(f'Estimated cooling time: {t_cool/3.15e7:.3f} yr')

        return t_cool

    def cool_step(self, dt):

        # find the amount of energy in envelope and shell

        inner_planet_mask = self.data['shell'] == 0
        outer_planet_mask = self.data['shell'] == 1
        shell_mask = self.data['shell'] == 2
        envelope_mask = self.data['shell'] == 3
        E_shell = np.nansum(np.abs(self.data['E'][shell_mask]))
        E_envelope = np.sum(self.data['E'][envelope_mask])

        E_outer = np.nansum(self.data['E'][outer_planet_mask])
        m_outer = np.nansum(self.data['m'][outer_planet_mask])
        u_outer = E_outer / m_outer
        self.data['u'][outer_planet_mask] = u_outer
        self.data['T'][outer_planet_mask] = fst.T2_EOS(self.data['u'][outer_planet_mask], self.data['rho'][outer_planet_mask])
        self.data['P'][outer_planet_mask] = fst.P_EOS(self.data['rho'][outer_planet_mask], self.data['T'][outer_planet_mask])
        self.data['s'][outer_planet_mask] = fst.P_EOS(self.data['rho'][outer_planet_mask], self.data['T'][outer_planet_mask])

        t_shell = E_shell / self.L_phot

    def analyse(self, plot_check=False):
        if plot_check:
            self.plot('rho', cmap='magma', limits=[-15, 15])
            self.plot('T', log=False, cmap='coolwarm')
            self.plot('alpha_v')
        self.initial_cool()
        self.remove_droplets()
        # self.hydrostatic_equilibrium(initial_extrapolation=False)
        self.get_photosphere()
        if plot_check:
            self.plot('rho', cmap='magma')
            self.plot('t_cool', plot_photosphere=True)
            self.plot('alpha', plot_photosphere=True)


if __name__ == '__main__':
    snap = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
    phot = photosphere(snap, resolution=1000, n_theta=80)
    phot.analyse(plot_check=False)
    phot.set_up_cooling_shells()
    phot.simple_cooling()
    phot.plot('rho', cmap='magma', plot_photosphere=True)
    phot.plot('T', cmap='inferno', log=False)

