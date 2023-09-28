# extracts data from a snapshot and analyses it, producing a 2D photosphere model
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import odeint
from scipy.special import exp1
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.slice import slice_gas
from unyt import Rearth, m
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
import uuid

from snapshot_analysis import snapshot, data_labels
import EOS as fst

cpus = cpu_count()

sigma = 5.670374419e-8
L_sun = 3.828e26
R_earth = 6371000
M_earth = 5.972e24
day = 3600 * 24
yr = 365.25 * day
silicate_latent_heat_v = 3e7
photosphere_depth = 2/3
pressure_shell = 1e20
outer_shell_depth = 1e-7
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


def replace_nan_with_nearest(arr):
    # Find indices of NaN values
    nan_indices = np.isnan(arr)

    # Create an array of coordinates for non-NaN values
    non_nan_indices = np.arange(len(arr))[~nan_indices]

    if len(non_nan_indices) == 0:
        # If there are no non-NaN values, we can't replace NaNs
        return arr

    # Interpolate using SciPy's interp1d to fill NaNs with the nearest non-NaN value
    interp_func = interp1d(non_nan_indices, arr[~nan_indices], kind='nearest', fill_value='extrapolate')
    interpolated_values = interp_func(np.arange(len(arr)))

    # Replace NaN values with the interpolated values
    arr[nan_indices] = interpolated_values[nan_indices]

    return arr


class photosphere:

    # sample size and max size both have units
    def __init__(self, snapshot, sample_size=12*Rearth, max_size=50*Rearth, resolution=500, n_theta=100, n_phi=10):

        self.j_phot = np.zeros(n_theta + 1)
        self.luminosity = 0
        self.T_photosphere, self.A_photosphere, self.R_photosphere = 0, 0, 0
        self.R_phot, self.z_phot = np.zeros(n_theta + 1), np.zeros(n_theta + 1)

        sample_size.convert_to_units(Rearth)
        max_size.convert_to_units(Rearth)
        self.snapshot = snapshot
        self.data = {}

        # calculates the indexes to sample from to fill the array
        r_range = np.linspace(0, sample_size.value * 0.95, num=int(resolution / 2)) * Rearth
        theta_range = np.arange(n_theta + 1) * (np.pi / n_theta)
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

        self.data['A_r-'] = 2 * pi * (r ** 2) * (cos(theta) - cos(theta + d_theta))
        self.data['A_r+'] = 2 * pi * ((r + dr) ** 2) * (cos(theta) - cos(theta + d_theta))

        self.data['A_theta-'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta)
        self.data['A_theta+'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta + d_theta)

        self.data['A_theta+'][-1, :] = np.zeros_like(self.data['A_theta+'][-1, :])

        self.data['V'] = (1/3) * pi * ((r + dr) ** 3 - r ** 3) * (cos(theta) - cos(theta + d_theta))

        # these values are used to calculate the index in the array for a given r and theta
        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[1] / max_size.value

        # values used to get the elliptical surface for the start of the extrapolation
        self.R_min, self.z_min = snapshot.HD_limit_R.value * 0.95, snapshot.HD_limit_z.value * 0.95
        # linear eccentricity of the extrapolation surface
        self.linear_eccentricity = np.sqrt(self.R_min ** 2 - self.z_min ** 2)

        self.central_mass = self.snapshot.total_mass.value
        self.data['test'] = self.data['h'] * (self.data['R'] ** -2)

        # R = np.linspace(0, 20) * R_earth
        # omega = self.snapshot.best_fit_rotation_curve_mks(R)
        # plt.plot(R / R_earth, omega)
        # plt.show()

        # self.data['omega'] = self.snapshot.best_fit_rotation_curve_mks(self.data['r'])
        # self.plot('omega', log=True)
        self.t_dyn = np.sqrt((max_size.value ** 3) / (6.674e-11 * self.central_mass))

        # extrapolation performed here
        self.entropy_extrapolation = self.extrapolate_entropy()
        self.hydrostatic_equilibrium(initial_extrapolation=True)

        self.calculate_EOS()

        self.verbose = True

    def plot(self, parameter, log=True, contours=None, cmap='turbo', plot_photosphere=False, limits=None, round_to=1, val_min=None, val_max=None):
        vals = np.log10(self.data[parameter]) if log else self.data[parameter]
        val_max = np.log10(val_max) if log and val_max is not None else val_max
        val_min = np.log10(val_min) if log and val_min is not None else val_min
        R, z = self.data['R'] * m, self.data['z'] * m
        R.convert_to_units(Rearth)
        z.convert_to_units(Rearth)

        plt.figure(figsize=(10, 8))
        # if val_max is not None:
        #     plt.contourf(R, z, vals, 200, cmap=cmap, vmax=val_max)
        # else:
        #     plt.contourf(R, z, vals, 200, cmap=cmap)

        cs = plt.contourf(R, z, vals, 200, cmap=cmap, norm=Normalize(vmin=val_min, vmax=val_max))

        cbar = plt.colorbar(mappable=cs, label=data_labels[parameter] if not log else '$\log_{10}$[' + data_labels[parameter] + ']')
        plt.xlabel(data_labels['R'])
        plt.ylabel(data_labels['z'])
        if limits is not None:
            plt.ylim(limits)
        
        if plot_photosphere:
            plt.plot(self.R_phot / R_earth, self.z_phot / R_earth, 'r--')
        theta = np.linspace(0, np.pi)
        plt.plot((self.R_min / R_earth) * np.sin(theta), (self.z_min / R_earth) * np.cos(theta), 'k--')

        vals = np.where(np.isfinite(vals), vals, np.NaN)
        min_tick = np.ceil(np.nanmin(vals / round_to))
        max_tick = np.ceil(np.nanmax(vals / round_to))
        try:
            tick_positions = np.arange(min_tick, max_tick) * round_to
            cbar.set_ticks(tick_positions)
        except ValueError:
            pass

        if contours is not None:
            cs = plt.contour(R, z, vals, contours, colors='black', linestyles='dashed')
            plt.clabel(cs, contours, colors='black')

        plt.show()

    def get_index(self, r, theta):
        i_r = np.int32(r * self.i_per_r)
        i_theta = np.int32(theta * self.i_per_theta)
        return i_theta, i_r

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

    def dPdr(self, P, r, theta, S_funct=None):
        gravity = - (6.674e-11 * self.central_mass) / (r ** 2)

        R = r * np.sin(theta)
        omega = self.snapshot.best_fit_rotation_curve_mks(r)
        centrifugal = R * (omega ** 2) * np.sin(theta)

        S = S_funct(r, theta)
        rho = fst.rho_EOS(S, P)

        result = rho * (gravity + centrifugal)
        return np.nan_to_num(result)

    def dlnPdr(self, lnP, r, theta, S_funct=None):
        gravity = - (6.674e-11 * self.central_mass) / (r ** 2)

        R = r * np.sin(theta)
        omega = self.snapshot.best_fit_rotation_curve_mks(r)
        centrifugal = R * (omega ** 2) * np.sin(theta)

        S = S_funct(r, theta)
        rho = fst.rho_EOS(S, np.exp(lnP))

        result = np.exp(-lnP) * rho * (gravity + centrifugal)
        return np.nan_to_num(result)

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

    def calculate_EOS(self):
        iron_mask = self.data['matid'] > 400.8

        self.data['alpha'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        self.data['alpha_v'] = fst.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)


        self.data['m'] = self.data['rho'] * self.data['V']
        self.data['E'] = self.data['u'] * self.data['m']
        self.data['rho_E'] = self.data['E'] / self.data['V']

        self.data['phase'] = fst.phase(self.data['s'], self.data['P'])
        self.data['vq'] = fst.vapor_quality(self.data['s'], self.data['P'])
        self.data['lvf'] = fst.liquid_volume_fraction(self.data['rho'], self.data['P'], self.data['s'])

    def remove_droplets(self, max_infall=1e4, check_infall=True, check_alpha=False):

        condensation_mask = self.data['phase'] == 2

        rho_drop = fst.rho_liquid(self.data['P'])
        rho_vapour = fst.rho_vapor(self.data['rho'], self.data['s'], self.data['P'])
        D0, CD = 1e-3, 0.5
        keplerian_omega = np.sqrt((6.674e-11 * self.central_mass) / (self.data['R'] ** 3))
        omega = self.snapshot.best_fit_rotation_curve_mks(self.data['R'])
        v = np.abs(self.data['R'] * (keplerian_omega - omega))

        t_infall = (2 * rho_drop * D0) / (rho_vapour * CD * v)
        t_infall = np.where(condensation_mask, t_infall, 0)
        self.data['t_infall'] = t_infall

        if check_infall:
            remove_mask = condensation_mask & (t_infall < max_infall)
        else:
            remove_mask = condensation_mask

        initial_mass = np.nansum(self.data['m'][remove_mask])
        new_S = fst.condensation_S(self.data['s'], self.data['P'])
        self.data['s'] = np.where(remove_mask, new_S, self.data['s'])
        self.data['rho'] = fst.rho_EOS(self.data['s'], self.data['P'])
        self.data['T'] = fst.T1_EOS(self.data['s'], self.data['P'])
        self.data['u'] = fst.u_EOS(self.data['rho'], self.data['T'])
        self.calculate_EOS()

        final_mass = np.nansum(self.data['m'][remove_mask])
        mass_lost = initial_mass - final_mass
        if self.verbose:
            print(f'Removing droplets: {mass_lost / M_earth:.2e} M_earth lost')

        if check_alpha:
            t = self.data['t_infall']
            V_photosphere = np.nansum(self.data['V'][self.data['tau'] > photosphere_depth])
            rho_drop = np.nanmean(fst.rho_liquid(self.data['P']))

            m_drop = t * (self.luminosity / silicate_latent_heat_v)
            lvf = m_drop / (rho_drop * V_photosphere)
            D0 = 1e-3
            alpha_drop = (6 / (4 * D0)) * lvf
            f = alpha_drop / self.data['alpha_v']
            self.data['test'] = np.where(self.data['tau'] > photosphere_depth, f, 0)
            self.plot('test', contours=[0])

        return mass_lost

    def get_photosphere(self, check_droplets=False):

        d_tau = self.data['alpha_v'] * self.data['dr']
        self.data['tau'] = np.flip(np.cumsum(np.flip(d_tau, axis=1), axis=1), axis=1)

        photosphere_mask = self.data['tau'] < photosphere_depth
        j_phot = np.argmax(photosphere_mask, axis=1)
        j_phot = np.where(j_phot == 0, self.n_r - 1, j_phot)
        i_phot = np.arange(self.n_theta)

        phot_indexes = tuple((i_phot, j_phot))
        # L = (self.data['A_r+'] * sigma * self.data['T'] ** 4)[phot_indexes]
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
            self.luminosity += F[i] * A

        if self.verbose:
            print(f'Photosphere found with luminosity {self.luminosity/L_sun:.2e} L_sun')

        self.T_photosphere = np.nanmean(self.data['T'][phot_indexes])
        self.R_photosphere = np.nanmean(self.data['r'][phot_indexes])

    def initial_cool(self, max_time):

        photosphere_mask = self.data['tau'] < photosphere_depth

        tau = self.data['alpha_v'] * self.data['dr']
        emissivity = 1 - np.exp(-tau) + tau * exp1(tau)
        L = sigma * self.data['T'] ** 4 * self.data['A_r+'] * emissivity
        t_cool = self.data['E'] / L

        # t_cool = np.where(~photosphere_mask, t_cool, 0)
        t_cool_total = np.flip(np.cumsum(np.flip(t_cool, axis=1), axis=1), axis=1)

        t_cool_per_dr = (t_cool / self.data['dr']) * R_earth

        self.data['test'] = t_cool_per_dr
        self.plot('test', plot_photosphere=True)

        max_time_mask = t_cool_total < max_time
        cool_mask = max_time_mask | photosphere_mask

        self.data['test'] = cool_mask * 1
        self.plot('test', log=False, plot_photosphere=True)

        r_cool = np.min(self.data['r'][max_time_mask])
        cool_mask = self.data['r'] > r_cool

        self.data['test'] = cool_mask * 1
        self.plot('test', plot_photosphere=True, log=False)

        u1, rho, T1 = self.data['u'], self.data['rho'], np.array(self.data['T'])
        cool_factor = np.minimum(max_time / t_cool, 0.75)
        du = cool_factor * u1
        u2 = u1 - du
        T2 = fst.T2_EOS(u2, rho)

        self.data['u'] = u2
        self.data['T'] = np.where(cool_mask, T2, T1)
        self.data['P'] = fst.P_EOS(rho, T2)
        self.data['s'] = fst.S_EOS(rho, T2)
        self.calculate_EOS()

    def cool_step(self, dt):

        photosphere_mask = self.data['tau'] > photosphere_depth
        pressure_mask = self.data['P'] < pressure_shell
        energy_mask = photosphere_mask & pressure_mask
        outer_mask = (self.data['tau'] > outer_shell_depth) & ~photosphere_mask

        u1 = self.data['u']
        P1 = self.data['P']

        # cool inner region
        m_in = np.sum(self.data['m'][energy_mask])
        E_in = np.sum(self.data['E'][energy_mask])
        E1 = np.array(self.data['E'])
        dE_in = self.luminosity * dt
        u_avg_in = E_in / m_in
        du_in = dE_in / m_in
        k = (1 - du_in / u_avg_in) if m_in > 0 else 1
        assert k <= 1
        k = np.maximum(k, 0.1)
        u2_in = u1 * k

        t_cool_estimated = E_in / self.luminosity

        # assert dt < t_cool_estimated or m_in == 0
        r = self.data['r']
        r_phot = self.R_photosphere
        equilibrium_temp = (0.5 ** 0.25) * ((r_phot / r) ** 0.5) * self.T_photosphere
        equilibrium_temp = np.where(~photosphere_mask & (r > r_phot), equilibrium_temp, np.nan)
        # test = np.sign(self.data['T'] - equilibrium_temp)
        # plt.imshow(test)
        # plt.show()
        # self.data['test'] = test
        # self.plot('test', log=False, round_to=1, plot_photosphere=True)

        self.data['u'] = np.where(pressure_mask, u2_in, u1)
        self.data['T'] = fst.T2_EOS(self.data['u'], self.data['rho'])
        self.data['P'] = fst.P_EOS(self.data['rho'], self.data['T'])
        self.data['S'] = fst.S_EOS(self.data['rho'], self.data['T'])
        self.calculate_EOS()
        self.get_photosphere()

        E2 = self.data['E']
        E_in_after = np.sum(self.data['E'][energy_mask])

        if E_in_after > E_in:
            mask = ((E1 > E2) & energy_mask)
            plt.scatter(u1[mask], P1[mask], s=5, c='red')
            plt.scatter(u1[~mask], P1[~mask], s=5, c='blue')
            plt.plot(fst.NewEOS.vc.Sv, fst.NewEOS.vc.Uv)
            plt.yscale('log')
            plt.show()
            self.nan_check()
            print(k)
            assert E_in_after < E_in

        if self.verbose:
            print(f'Cooling by {du_in / u_avg_in:.3%} over {dt / (3600 * 24):.2f} days')
            print(f'Energy loss inner region: {dE_in:.2e} ({du_in / u_avg_in:.3%})')
        # print(f'Energy loss outer region: {dE_out:.2e} ({du_out / u_avg_out:.3%})')

    def initial_cool_v2(self, max_time):

        u1, rho, T1 = self.data['u'], self.data['rho'], np.array(self.data['T'])
        tau = self.data['alpha_v'] * R_earth

        emissivity = 1 - np.exp(-tau) + tau * exp1(tau)
        F = sigma * T1 ** 4 * emissivity
        t_cool = ((u1 * rho) / F) * R_earth

        min_cooling_time = np.nanmin(t_cool)
        # dt = min_cooling_time * 0.9

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

        # u1, rho, T1 = self.data['u'], self.data['rho'], np.array(self.data['T'])
        # tau = self.data['alpha_v'] * R_earth
        #
        # emissivity = 1 - np.exp(-tau) + tau * exp1(tau)
        # F = sigma * T1 ** 4 * emissivity
        # t_cool = ((u1 * rho) / F) * R_earth
        #
        # min_cooling_time = np.nanmin(t_cool)
        # print(f'MIN: {min_cooling_time:.2e}')
        #
        # self.data['t_cool'] = t_cool
        # self.plot('t_cool')

        return True

    def long_term_evolution(self, total_time=40, div=40, plot=False, plot_interval=10):

        self.verbose = False

        total_mass_loss = 0
        L = [self.luminosity]
        A = [self.A_photosphere]
        R = [self.R_photosphere]
        T = [self.T_photosphere]
        t = [0]

        E_in = np.sum(self.data['E'][(self.data['tau'] > photosphere_depth) & (self.data['P'] < pressure_shell)])
        t_cool_estimated = E_in / self.luminosity
        print(f'Estimated cooling time: {t_cool_estimated:.1e} seconds ({t_cool_estimated / yr:.2f} years)')
        dt = t_cool_estimated / div
        t_max = total_time * yr
        n = int(t_max / dt)

        print(f'Cooling for {n * dt:.1e} seconds ({n * dt / yr:.2f} years):')

        for i in tqdm(range(1, n + 1)):
            self.nan_check()
            self.cool_step(dt)
            total_mass_loss += self.remove_droplets()
            #self.nan_check()

            if i % plot_interval == 0 and plot:
                self.plot('rho', plot_photosphere=True, val_min=1e-4)
                self.plot('T', log=False, round_to=1000, val_max=4000, plot_photosphere=True)

            t.append(i * dt)
            L.append(self.luminosity)
            A.append(self.A_photosphere)
            R.append(self.R_photosphere)
            T.append(self.T_photosphere)

        t = np.array(t)
        L, A, R, T = np.array(L), np.array(A), np.array(R), np.array(T)

        i_half = np.argmin((L / L[0]) > 0.5)
        i_tenth = np.argmin((L / L[0]) > 0.1)
        t_half = t[i_half]
        t_tenth = t[i_tenth]

        self.verbose = True

        return t, L, A, R, T, total_mass_loss, t_half, t_tenth

    def set_up(self):

        self.initial_cool_v2(1e5)
        self.nan_check()
        self.remove_droplets()

        self.get_photosphere()

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

                # nan_mask = np.isnan(self.data[k])
                # print(f'NaNs found in {k}')
                # self.data['test'] = nan_mask * 1
                # self.plot('test', log=False, cmap='cubehelix')

                # for k2 in self.data.keys():
                #     print(f'{k2}: {self.data[k2][nan_mask]}')

    def nan_fix(self):

        nan_mask = np.isnan(self.data['rho']) | np.isnan(self.data['T'])

        nan_rho = 1e-5
        nan_T = 2200
        nan_P = fst.P_EOS(nan_rho, nan_T)
        nan_S = fst.S_EOS(nan_rho, nan_T)
        nan_u = fst.u_EOS(nan_rho, nan_T)

        self.data['rho'] = np.where(nan_mask, nan_rho, self.data['rho'])
        self.data['T'] = np.where(nan_mask, nan_T, self.data['T'])
        self.data['P'] = np.where(nan_mask, nan_P, self.data['P'])
        self.data['s'] = np.where(nan_mask, nan_S, self.data['s'])
        self.data['u'] = np.where(nan_mask, nan_u, self.data['u'])


if __name__ == '__main__':
    snap = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
    phot = photosphere(snap, n_theta=50)
    phot.set_up()

    t1, L1, A1, R1, T1, m1 = phot.long_term_evolution(plot=False, plot_interval=40)

    phot2 = photosphere(snap, n_theta=50)
    phot2.set_up()
    t2, L2, A2, R2, T2, m2 = phot2.long_term_evolution(plot=False, plot_interval=40)

    plt.plot(t1 / yr, L1 / L_sun)
    plt.plot(t2 / yr, L2 / L_sun)
    plt.show()

    plt.plot(t1 / yr, R1 / R_earth)
    #plt.plot(t2 / yr, R2 / R_earth)
    plt.show()

    plt.plot(t1 / yr, T1)
    #plt.plot(t2 / yr, T2)
    plt.show()

# new sims that work: 0, 1, 3, 4, 6, 8
# 8, 9 requires cooling
# 4, 6 has some weird bits
# 5 also (not really a synestia)
