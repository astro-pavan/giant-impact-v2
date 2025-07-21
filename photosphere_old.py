# extracts data from a snapshot and analyses it, producing a 2D photosphere model
# all values are in SI units unless otherwise specified

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

from snapshot_analysis import data_labels
import EOS as fst

cpus = cpu_count()

# constants and function definitions
sigma = 5.670374419e-8 # stefan-boltzmann constant
L_sun = 3.828e26
R_earth = 6371000
M_earth = 5.972e24
day = 3600 * 24
yr = 365.25 * day
silicate_latent_heat_v = 3e7
photosphere_depth = 2/3
pi = np.pi
cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)


# allows a function defined within another function to be used in a multiprocessing pool
def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)

    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


# calculates v in elliptical coordinates
def get_v(R, z, a):
    v = np.sign(z) * np.arccos((np.sqrt((R + a) ** 2 + z ** 2) - np.sqrt((R - a) ** 2 + z ** 2)) / (2 * a))
    return np.nan_to_num(v)


# class containing the photosphere model
class photosphere:

    # sample size and max size both have units
    def __init__(self, snapshot, sample_size=12*Rearth, max_size=50*Rearth, period=None,
                 resolution=500, n_theta=100, n_phi=10, droplet_infall=True, pressure_limit=1e11):

        sample_size.convert_to_units(Rearth)
        max_size.convert_to_units(Rearth)
        self.snapshot = snapshot
        self.data = {}
        self.droplet_infall = droplet_infall
        self.pressure_shell = pressure_limit

        self.j_phot = np.zeros(n_theta+1)
        self.luminosity = 0
        self.T_photosphere, self.A_photosphere, self.R_photosphere = 0, 0, 0
        self.R_phot, self.z_phot = np.zeros(n_theta+1), np.zeros(n_theta+1)

        self.central_mass = self.snapshot.total_mass

        if period is not None:
            G = 6.67430e-11
            max_size_mks = ((G * self.central_mass * period * period) / (12 * np.pi * np.pi)) ** (1 / 3)
            max_size = (max_size_mks / 6371000) * Rearth
            print(f'Hill radius = {max_size}')

        if max_size < sample_size:
            sample_size = max_size

        # calculates the indexes to sample from to fill the array
        r_range = np.linspace(0, sample_size.value * 0.95, num=int(resolution / 2)) * Rearth
        theta_range = np.arange(n_theta+1) * (np.pi / n_theta)
        r, theta = np.meshgrid(r_range, theta_range)
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
            matids = get_slice('material_ids')

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
            data['matid'] = matids

            return data

        print('Loading data into photosphere model:')

        # loads multiple sections at different phi angles and averages them
        self.data = get_section(0)

        for i in tqdm(range(1, n_phi)):
            vals = get_section(np.pi / n_phi * i)
            for k in self.data.keys():
                self.data[k] = (i * self.data[k] + vals[k]) / (i + 1)

        # fixes an error with infinite pressure
        infinite_mask = np.isfinite(self.data['P'])
        P_fix = fst.P_EOS(self.data['rho'], self.data['T'].value)
        self.data['P'] = np.where(infinite_mask, self.data['P'], P_fix)

        max_size.convert_to_mks()

        # extends the data arrays ready for extrapolation
        for k in self.data.keys():
            if k == 'r':
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'linear_ramp', end_values=(0, max_size.value))
            else:
                self.data[k] = np.pad(self.data[k], ((0, 0), (0, extend_r)), 'edge' if k == 'theta' or k == 'matid' else 'constant')

        self.n_r, self.n_theta = self.data['r'].shape[1], self.data['r'].shape[0]

        # calculates the R and z coordinates for each point
        self.data['R'] = self.data['r'] * np.sin(self.data['theta'])
        self.data['z'] = self.data['r'] * np.cos(self.data['theta'])

        self.data['dr'] = np.roll(self.data['r'], -1, axis=1) - self.data['r']
        self.data['dr'][:, -1] = self.data['dr'][:, -2]
        self.data['d_theta'] = np.full_like(self.data['dr'], np.pi / n_theta)

        r, dr = self.data['r'], self.data['dr']
        theta, d_theta = self.data['theta'], self.data['d_theta']
        self.data['A'] = 2 * pi * ((r + dr) ** 2) * (cos(theta) - cos(theta + d_theta))
        self.data['V'] = (1 / 3) * pi * ((r + dr) ** 3 - r ** 3) * (cos(theta) - cos(theta + d_theta))
        self.data['V'] = np.abs(self.data['V'])

        # these values are used to calculate the index in the array for a given r and theta
        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[1] / max_size.value

        # values used to get the elliptical surface for the start of the extrapolation
        self.R_min, self.z_min = snapshot.HD_limit_R.value * 0.95, snapshot.HD_limit_z.value * 0.95
        # linear eccentricity of the extrapolation surface
        self.linear_eccentricity = np.sqrt(self.R_min ** 2 - self.z_min ** 2)

        self.data['test'] = self.data['h'] * (self.data['R'] ** -2)

        # extrapolation performed here
        if sample_size < max_size:
            self.entropy_extrapolation = self.extrapolate_entropy()
            self.hydrostatic_equilibrium(initial_extrapolation=True)
        else:
            self.data['u'] = fst.u_EOS(self.data['rho'], self.data['T'])

        self.calculate_EOS()

        self.verbose = True

    # plots a cross-section of the photosphere as a contour plot for a given parameter
    def plot(self, parameter, log=True, contours=None, cmap='turbo', plot_photosphere=False, round_to=1,
             val_min=None, val_max=None, save=None, xlim=None, ylim=None):
        vals = np.log10(self.data[parameter]) if log else self.data[parameter]

        if val_min is None:
            val_min = np.nanmin(vals)
        elif log:
            val_min = np.log10(val_min)

        if val_max is None:
            val_max = np.nanmax(vals)
        elif log:
            val_max = np.log10(val_max)

        R, z = self.data['R'] * m, self.data['z'] * m
        R.convert_to_units(Rearth)
        z.convert_to_units(Rearth)

        plt.figure(figsize=(8, 6), dpi=300)
        # if val_max is not None:
        #     plt.contourf(R, z, vals, 200, cmap=cmap, vmax=val_max)
        # else:
        #     plt.contourf(R, z, vals, 200, cmap=cmap)

        levels = np.linspace(val_min, val_max, 200)

        cs = plt.contourf(R, z, vals, levels, cmap=cmap, extend='both')

        cbar = plt.colorbar(mappable=cs, label=data_labels[parameter] if not log else '$\log_{10}$[' + data_labels[parameter] + ']')
        plt.xlabel(data_labels['R'])
        plt.ylabel(data_labels['z'])

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        
        if plot_photosphere:
            plt.plot(self.R_phot / R_earth, self.z_phot / R_earth, 'r--', label='Surface of photosphere')
        theta = np.linspace(0, np.pi)
        plt.plot((self.R_min / R_earth) * np.sin(theta), (self.z_min / R_earth) * np.cos(theta), 'w--', label='Extrapolation point')

        plt.legend()

        vals = np.where(np.isfinite(vals), vals, np.NaN)
        min_tick = np.ceil(val_min / round_to)
        max_tick = np.ceil(val_max / round_to)
        try:
            tick_positions = np.arange(min_tick, max_tick) * round_to
            cbar.set_ticks(tick_positions)
        except ValueError:
            pass

        if contours is not None:
            cs = plt.contour(R, z, vals, contours, colors='black', linestyles='dashed')
            plt.clabel(cs, contours, colors='black')

        if save is not None:
            plt.savefig(f'figures/{save}.pdf', bbox_inches='tight')
            plt.savefig(f'figures/{save}.png', bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    # gets the index in the data array for a given r and theta
    def get_index(self, r, theta):
        i_r = np.int32(r * self.i_per_r)
        i_theta = np.int32(theta * self.i_per_theta)
        return i_theta, i_r

    # extrapolates the entropy at a constant value using elliptical coodinates
    def extrapolate_entropy(self):

        print(f'Extrapolating from R = {self.R_min / R_earth:.2f}, z = {self.z_min / R_earth:.2f}')

        n_v = 400
        v = - (np.arange(n_v) * (np.pi / n_v) - np.pi / 2)
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

    # gradient of log(pressure)
    def dlnPdr(self, lnP, r, theta, S_funct=None):
        gravity = - (6.674e-11 * self.central_mass) / (r ** 2)

        R = r * np.sin(theta)
        omega = self.snapshot.best_fit_rotation_curve_mks(r)
        centrifugal = R * (omega ** 2) * np.sin(theta)

        S = S_funct(r, theta)
        rho = fst.rho_EOS(S, np.exp(lnP))

        result = np.exp(-lnP) * rho * (gravity + centrifugal)
        return np.nan_to_num(result)

    # extrapolates to the outer regions using the rotating hydrostatic equilibrium model
    def hydrostatic_equilibrium(self, initial_extrapolation=False):

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

            r_0 = np.sqrt((2 * np.sin(theta)) ** 2 + (2 * np.cos(theta)) ** 2) * R_earth
            j_start = self.get_index(r_0, theta)[1]

        @globalize
        def extrapolate(i):

            j_0 = np.int32(j_start[i])

            # P_0 = self.data['P'][i, j_0]
            # f = lambda P, r: self.dPdr(P, r, theta[i], S_funct=S_funct)

            P_0 = np.log(self.data['P'][i, j_0])
            f = lambda P, r: self.dlnPdr(P, r, theta[i], S_funct=S_funct)

            r_solution = self.data['r'][i, j_0:]
            solution = odeint(f, P_0, r_solution)

            # nan fix
            solution = np.where(np.isnan(solution), 4e-4, solution)

            # P_solution = np.nan_to_num(solution)
            P_solution = np.exp(np.nan_to_num(solution))

            print(u"\u2588", end='')
            return P_solution.T, j_0, i

        pool = Pool(cpus - 1)
        results = pool.map(extrapolate, range(self.n_theta))
        print(' DONE')

        for r in results:
            i, j_0 = r[2], r[1]
            self.data['P'][i:i + 1, j_0:] = r[0]

        self.data['rho'] = fst.rho_EOS(self.data['s'], self.data['P'])
        self.data['T'] = fst.T1_EOS(self.data['s'], self.data['P'])
        self.data['u'] = fst.u_EOS(self.data['rho'], self.data['T'])

    # updates the alpha and other thermodynamic variables (run once rho, T, P, S have been updated)
    def calculate_EOS(self):

        self.data['alpha'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        self.data['alpha_v'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)

        self.data['m'] = self.data['rho'] * self.data['V']
        self.data['E'] = self.data['u'] * self.data['m']
        self.data['rho_E'] = self.data['E'] / self.data['V']

        self.data['phase'] = fst.phase(self.data['s'], self.data['P'])
        self.data['vq'] = fst.vapor_quality(self.data['s'], self.data['P'])
        self.data['lvf'] = fst.liquid_volume_fraction(self.data['rho'], self.data['P'], self.data['s'])

    # removes droplets that have condensed
    def remove_droplets(self, max_infall=1e4, check_infall=True, check_alpha=False, dt=1):

        condensation_mask = self.data['phase'] == 2

        rho_drop = fst.rho_liquid(self.data['P'])
        rho_vapour = fst.rho_vapor(self.data['rho'], self.data['s'], self.data['P'])
        D0, CD = 1e-3, 0.5
        keplerian_omega = np.sqrt((6.674e-11 * self.central_mass) / (self.data['R'] ** 3))
        omega = self.snapshot.best_fit_rotation_curve_mks(self.data['R'])
        v_rel = np.abs(self.data['R'] * (keplerian_omega - omega))
        v_orb = np.abs(self.data['R'] * keplerian_omega)

        t_infall = (2 * rho_drop * D0 * v_orb) / (rho_vapour * CD * (v_rel ** 2))
        t_infall = np.where(condensation_mask, t_infall, 0)
        self.data['t_infall'] = t_infall

        if check_infall:
            remove_mask = condensation_mask & (t_infall < max_infall)
        else:
            remove_mask = condensation_mask

        initial_mass = np.array(self.data['m'])
        total_initial_mass = np.nansum(initial_mass[remove_mask])
        new_S = fst.condensation_S(self.data['s'], self.data['P'])
        self.data['s'] = np.where(remove_mask, new_S, self.data['s'])
        self.data['rho'] = fst.rho_EOS(self.data['s'], self.data['P'])
        self.data['T'] = np.nan_to_num(fst.T1_EOS(self.data['s'], self.data['P']))
        self.data['u'] = fst.u_EOS(self.data['rho'], self.data['T'])
        self.calculate_EOS()

        final_mass = self.data['m']
        total_final_mass = np.nansum(final_mass[remove_mask])
        mass_lost = total_initial_mass - total_final_mass
        if self.verbose:
            print(f'Removing droplets: {mass_lost / M_earth:.2e} M_earth lost')

        if check_alpha:
            phot_mask = self.data['tau'] > photosphere_depth

            droplet_volume = ((initial_mass - final_mass) * (t_infall / dt)) / rho_drop
            lvf = droplet_volume / self.data['V']
            alpha_drop = (6 / (4 * D0)) * lvf

            alpha_drop = np.where(condensation_mask, alpha_drop, 0)

            alpha_drop_mean = np.sum(alpha_drop * self.data['V']) / np.sum(self.data['V'][condensation_mask])
            if alpha_drop_mean > 1e-8 and self.verbose:
                print(alpha_drop_mean)

        self.get_photosphere()

        return mass_lost

    # calculates the photosphere and luminonsity of the post impact body
    def get_photosphere(self):

        d_tau = self.data['alpha_v'] * self.data['dr']
        self.data['tau'] = np.flip(np.cumsum(np.flip(d_tau, axis=1), axis=1), axis=1)

        photosphere_mask = self.data['tau'] < photosphere_depth
        j_phot = np.argmax(photosphere_mask, axis=1)
        j_phot = np.where(j_phot == 0, self.n_r - 1, j_phot)
        i_phot = np.arange(self.n_theta)

        phot_indexes = tuple((i_phot, j_phot))
        # L = (self.data['A'] * sigma * self.data['T'] ** 4)[phot_indexes]
        self.luminosity, self.A_photosphere = 0, 0
        F = (sigma * self.data['T'] ** 4)[phot_indexes]
        self.R_phot = self.data['R'][phot_indexes]
        self.z_phot = self.data['z'][phot_indexes]

        for i in range(len(self.R_phot) - 1):
            R1, R2 = self.R_phot[i], self.R_phot[i + 1]
            z1, z2 = self.z_phot[i], self.z_phot[i + 1]
            m = (z2 - z1)/(R2 - R1)
            A = np.abs(pi * np.sqrt(1 + m ** 2) * (R2 ** 2 - R1 ** 2))
            self.A_photosphere += A
            if np.isnan(F[i]):
                print(i)
            self.luminosity += F[i] * A

        if self.verbose:
            print(f'Photosphere found with luminosity {self.luminosity/L_sun:.2e} L_sun')

        self.T_photosphere = np.nanmean(self.data['T'][phot_indexes])
        self.R_photosphere = np.nanmean(self.data['r'][phot_indexes])

    # initially cools the post impact body to account for cooling not performed by SWIFT
    def initial_cool(self, max_time):

        u1, rho, T1 = self.data['u'], self.data['rho'], np.array(self.data['T'])
        alpha = self.data['alpha']
        A = self.data['A']
        V = self.data['V']
        L = V / A

        emissivity = 1 - np.exp(-alpha * L)
        t_cool = (rho * u1 * L) / (sigma * (T1 ** 4) * emissivity)

        t_cool = np.flip(np.cumsum(np.flip(t_cool, axis=1), axis=1), axis=1)

        min_cooling_time = np.nanmin(t_cool)

        if min_cooling_time > max_time:
            if self.verbose:
                print('Max initial cooling time exceeded')
            return False

        if self.verbose:
            print(f'Initial cool for {min_cooling_time:.1e} seconds')

        k = np.minimum(max_time / t_cool, 0.5)
        du = k * u1
        u2 = u1 - du
        T2 = fst.T2_EOS(u2, rho)

        self.data['u'] = u2
        self.data['T'] = T2
        self.data['P'] = fst.P_EOS(rho, T2)
        self.data['s'] = fst.S_EOS(rho, T2)
        self.calculate_EOS()

        return True

    # cools the post impact body for a time dt (assumes constant luminosity)
    def cool_step(self, dt):

        photosphere_mask = self.data['tau'] > photosphere_depth
        pressure_mask = self.data['P'] < self.pressure_shell
        energy_mask = photosphere_mask & pressure_mask

        u1 = self.data['u']

        # cool inner region
        m_in = np.sum(self.data['m'][energy_mask])
        E_in = np.sum(self.data['E'][energy_mask])
        dE_in = self.luminosity * dt
        u_avg_in = E_in / m_in
        du_in = dE_in / m_in
        k = (1 - du_in / u_avg_in) if m_in > 0 else 1

        u2_in = u1 * k
        u2 = np.where(pressure_mask, u2_in, u1)

        assert k <= 1

        self.data['u'] = u2
        self.data['T'] = np.nan_to_num(fst.T2_EOS(self.data['u'], self.data['rho']))
        self.data['P'] = fst.P_EOS(self.data['rho'], self.data['T'])
        self.data['S'] = fst.S_EOS(self.data['rho'], self.data['T'])
        self.calculate_EOS()
        self.get_photosphere()

        if self.verbose:
            print(f'Cooling by {du_in / u_avg_in:.3%} over {dt / (3600 * 24):.2f} days')
            print(f'Energy loss inner region: {dE_in:.2e} ({du_in / u_avg_in:.3%})')

    # cools the post impact body for a longer period (times are given in years)
    def long_term_evolution(self, max_time=100, max_count=1000,
                            plot=False, plot_interval=1, save_name='impact', plot_max=20):

        print('Cooling...')

        self.verbose = False

        self.cool_step(1e5)
        if self.droplet_infall:
            self.remove_droplets()

        t = [0]
        L = [self.luminosity]
        A = [self.A_photosphere]
        R = [self.R_photosphere]
        T = [self.T_photosphere]
        m_dot = [0]

        i = 0
        t_current = 0
        t_plot = 0
        plot_count = 0

        min_timestep = 0.015

        while t[i] < max_time * yr and i < max_count:

            i += 1

            E_in = np.sum(self.data['E'][(self.data['tau'] > photosphere_depth) & (self.data['P'] < self.pressure_shell)])
            t_cool_estimated = E_in / self.luminosity

            dt = (t_cool_estimated / 50) if t_current > 1 * yr else min_timestep * yr
            t_current += dt
            t_plot += dt

            self.nan_check()
            if np.isnan(self.luminosity):
                plt.plot(np.array(t), np.array(L) / L_sun)
                plt.show()
                plt.plot(t, A)
                plt.show()
                plt.plot(t, T)
                plt.show()
                self.plot('tau', log=True, val_min=1e0, val_max=1e10, plot_photosphere=True)
                break
            else:
                self.cool_step(dt)

            if self.droplet_infall:
                mass_loss = self.remove_droplets(dt=dt)
            else:
                mass_loss = 0

            t.append(t_current)
            L.append(self.luminosity)
            A.append(self.A_photosphere)
            R.append(self.R_photosphere)
            T.append(self.T_photosphere)
            m_dot.append(mass_loss / dt)

            if t_plot > plot_interval * yr and plot and plot_count < plot_max:
                self.plot('rho', plot_photosphere=True, val_max=1e4, val_min=1e-4, ylim=[-20, 20], xlim=[0, 40],
                          save=f'cooling/{save_name}_rho_t{t_current / yr:2.2f}', cmap='magma')
                self.plot('T', log=False, round_to=500, val_min=1000, val_max=4000, plot_photosphere=True, ylim=[-25, 25], xlim=[0, 50],
                          save=f'cooling/{save_name}_T_t{t_current / yr:2.2f}', cmap='inferno')
                t_plot = 0
                plot_count += 1

        self.verbose = True

        print('Cooling complete')

        t = np.array(t)
        L, A, R, T = np.array(L), np.array(A), np.array(R), np.array(T)

        i_half = np.argmin((L / L[0]) > 0.5)
        i_tenth = np.argmin((L / L[0]) > 0.1)
        t_half = t[i_half]
        t_tenth = t[i_tenth]

        return t, L, A, R, T, m_dot, t_half, t_tenth

    # performs the initial cooling, removes droplets and calculates the photosphere
    def set_up(self, extra_cool=None):

        self.initial_cool(1e5)
        self.nan_check()
        if self.droplet_infall:
            self.remove_droplets()
        if extra_cool is not None:
            self.cool_step(extra_cool)

        self.get_photosphere()

    # checks for any NaNs in the data and replaces the NaNs with nearby values if any are found
    def nan_check(self):

        if self.verbose:
            print('Checking for NaNs')

        for k in ['rho', 'T', 'P', 's', 'u']:

            if np.any(np.isnan(self.data[k])):

                for i in range(self.n_theta):
                    for j in range(self.n_r):
                        if np.isnan(self.data[k][i, j]):
                            self.data[k][i, j] = self.data[k][i, j - 1]

        self.data['m'] = self.data['rho'] * self.data['V']
        self.data['E'] = self.data['u'] * self.data['m']

