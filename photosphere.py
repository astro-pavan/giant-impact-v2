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
import EOS

cpus = cpu_count()

sigma = 5.670374419e-8
L_sun = 3.828e26
R_earth = 6371000
M_earth = 5.972e24
yr = 3.15e7
day = 3600 * 24
photosphere_depth = 2/3
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


class photosphere:

    # sample size and max size both have units
    def __init__(self, snapshot, sample_size=12*Rearth, max_size=40*Rearth, resolution=500, n_theta=100, n_phi=10):

        self.j_phot = np.zeros(n_theta + 1)
        self.luminosity = 0
        self.R_phot, self.z_phot = np.zeros(n_theta + 1), np.zeros(n_theta + 1)

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
        # infinite_mask = np.isfinite(self.data['P'])
        # P_fix = EOS.P_fst_EOS(self.data['rho'], self.data['T'].value)
        # self.data['P'] = np.where(infinite_mask, self.data['P'], P_fix)

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

        self.data['A_r+'] = 2 * pi * ((r + dr) ** 2) * (cos(theta) - cos(theta + d_theta))
        self.data['V'] = (1 / 3) * pi * ((r + dr) ** 3 - r ** 3) * (cos(theta) - cos(theta + d_theta))
        # self.data['A_r-'] = 2 * pi * (r ** 2) * (cos(theta) - cos(theta + d_theta))
        # self.data['A_theta-'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta)
        # self.data['A_theta+'] = pi * ((r + dr) ** 2 - r ** 2) * sin(theta + d_theta)
        # self.data['A_theta+'][-1, :] = np.zeros_like(self.data['A_theta+'][-1, :])

        # these values are used to calculate the index in the array for a given r and theta
        self.i_per_theta = n_theta / np.pi
        self.i_per_r = self.data['r'].shape[1] / max_size.value

        # values used to get the elliptical surface for the start of the extrapolation
        self.R_min, self.z_min = snapshot.HD_limit_R.value * 0.95, snapshot.HD_limit_z.value * 0.95
        # linear eccentricity of the extrapolation surface
        self.linear_eccentricity = np.sqrt(self.R_min ** 2 - self.z_min ** 2)

        self.central_mass = self.snapshot.total_mass.value
        self.data['omega'] = self.data['h'] * (self.data['R'] ** -2)
        self.iron_mask = self.data['matid'] > 400.8

        # extrapolation performed here
        self.entropy_extrapolation = self.extrapolate_entropy()
        self.hydrostatic_equilibrium(initial_extrapolation=True)
        self.calculate_EOS()

        self.verbose = True

    def plot(self, parameter, log=True, contours=None, cmap='turbo', plot_photosphere=False, limits=None, round_to=1, val_max=None):
        vals = np.log10(self.data[parameter]) if log else self.data[parameter]
        val_max = np.log10(val_max) if log and val_max is not None else val_max
        R, z = self.data['R'] * m, self.data['z'] * m
        R.convert_to_units(Rearth)
        z.convert_to_units(Rearth)

        plt.figure(figsize=(10, 8))
        # if val_max is not None:
        #     plt.contourf(R, z, vals, 200, cmap=cmap, vmax=val_max)
        # else:
        #     plt.contourf(R, z, vals, 200, cmap=cmap)

        cax = plt.contourf(R, z, vals, 200, cmap=cmap, norm=Normalize(vmax=val_max))

        cbar = plt.colorbar(cax, label=data_labels[parameter] if not log else '$\log_{10}$[' + data_labels[parameter] + ']')
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
        tick_positions = np.arange(min_tick, max_tick) * round_to
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

        print(f'Extrapolating from R = {self.R_min / R_earth:.2f}, z = {self.z_min / R_earth:.2f}')

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
        rho = EOS.rho_fst_EOS(S, P)

        result = rho * (gravity + centrifugal)
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
            S_funct = lambda x, y: S_interp(EOS.make_into_pair_array(y, x))

            r_0 = np.sqrt((2 * np.sin(theta)) ** 2 + (2 * np.cos(theta)) ** 2) * R_earth
            j_start = self.get_index(r_0, theta)[1]

        @globalize
        def extrapolate(i):

            j_0 = np.int32(j_start[i])
            P_0 = self.data['P'][i, j_0]

            f = lambda P, r: self.dPdr(P, r, theta[i], S_funct=S_funct)

            r_solution = self.data['r'][i, j_0:]
            solution = odeint(f, P_0, r_solution)
            #solution = solve_ivp(lambda t, y: f(y, t), )

            P_solution = np.nan_to_num(solution)

            print(u"\u2588", end='')
            return P_solution.T, j_0, i

        pool = Pool(cpus - 1)
        results = pool.map(extrapolate, range(self.n_theta))
        print(' DONE')

        for r in results:
            i, j_0 = r[2], r[1]
            self.data['P'][i:i + 1, j_0:] = r[0]

        self.data['rho'] = EOS.rho_fst_EOS(self.data['s'], self.data['P'])
        self.data['T'] = EOS.T1_fst_EOS(self.data['s'], self.data['P'])

    def calculate_EOS(self):

        alpha = EOS.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'])
        alpha_v = EOS.alpha(self.data['rho'], self.data['T'], self.data['P'], self.data['s'], D0=0)

        self.data['alpha'] = alpha
        self.data['alpha_v'] = alpha_v

        # self.data['u'] = np.where(self.iron_mask, EOS.u_iron_EOS(self.data['rho'], self.data['T']), EOS.u_fst_EOS(self.data['rho'], self.data['T']))
        self.data['u'] = EOS.u_fst_EOS(self.data['rho'], self.data['T'])
        self.data['m'] = self.data['rho'] * self.data['V']
        self.data['E'] = self.data['u'] * self.data['m']
        self.data['rho_E'] = self.data['E'] / self.data['V']

        self.data['phase'] = EOS.phase(self.data['s'], self.data['P'])
        # self.data['vq'] = EOS.vapor_quality(self.data['s'], self.data['P'])
        # self.data['lvf'] = EOS.liquid_volume_fraction(self.data['rho'], self.data['P'], self.data['s'])

    def remove_droplets(self):
        condensation_mask = self.data['phase'] == 2
        initial_mass = np.nansum(self.data['m'][condensation_mask])

        new_S = EOS.condensation_S(self.data['s'], self.data['P'])
        self.data['s'] = np.where(condensation_mask, new_S, self.data['s'])
        self.data['rho'] = EOS.rho_fst_EOS(self.data['s'], self.data['P'])
        self.data['T'] = EOS.T1_fst_EOS(self.data['s'], self.data['P'])
        self.calculate_EOS()

        final_mass = np.nansum(self.data['m'][condensation_mask])
        mass_lost = initial_mass - final_mass
        if self.verbose:
            print(f'Removing droplets: {mass_lost / M_earth:.2e} M_earth lost')

        return mass_lost

    def get_photosphere(self):

        d_tau = self.data['alpha_v'] * self.data['dr']
        self.data['tau'] = np.flip(np.cumsum(np.flip(d_tau, axis=1), axis=1), axis=1)

        photosphere_mask = self.data['tau'] < photosphere_depth
        j_phot = np.argmax(photosphere_mask, axis=1)
        j_phot = np.where(j_phot == 0, self.n_r - 1, j_phot)
        i_phot = np.arange(self.n_theta)

        phot_indexes = tuple((i_phot, j_phot))
        L = (self.data['A_r+'] * sigma * self.data['T'] ** 4)[phot_indexes]
        self.R_phot = self.data['R'][phot_indexes]
        self.z_phot = self.data['z'][phot_indexes]
        self.luminosity = np.sum(L)

        if self.verbose:
            print(f'Photosphere found with luminosity {self.luminosity/L_sun:.2e} L_sun')

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
        T2 = EOS.T2_fst_EOS(u2, rho)

        self.data['T'] = np.where(cool_mask, T2, T1)
        self.data['P'] = EOS.P_fst_EOS(rho, T2)
        self.data['s'] = EOS.S_fst_EOS(rho, T2)
        self.calculate_EOS()

    def cool_step(self, dt):

        photosphere_mask = self.data['tau'] > photosphere_depth
        outer_mask = (self.data['tau'] > outer_shell_depth) & ~photosphere_mask

        u1 = self.data['u']

        # cool inner region
        m_in = np.nansum(self.data['m'][photosphere_mask])
        E_in = np.nansum(self.data['E'][photosphere_mask])
        dE_in = self.luminosity * dt
        u_avg_in = E_in / m_in
        du_in = dE_in / m_in
        u2 = u1 * (1 - du_in / u_avg_in)

        T2 = EOS.T2_fst_EOS(u2, self.data['rho'])

        self.data['T'] = T2
        self.data['P'] = EOS.P_fst_EOS(self.data['rho'], T2)
        self.data['s'] = EOS.S_fst_EOS(self.data['rho'], T2)
        self.calculate_EOS()

        # # cool outer region
        # m_out = np.nansum(self.data['m'][outer_mask])
        # E_out = np.nansum(self.data['E'][outer_mask])
        # dE_out = self.luminosity * np.exp(outer_shell_depth - photosphere_depth) * dt
        # u_avg_out = E_out / m_out
        # du_out = dE_out / m_out
        # u2_out = u1 * (1 - du_out / u_avg_out)

        # u2 = np.where(photosphere_mask, u2_in, u1)
        # u2 = np.where(outer_mask, u2_in, u1)

        # self.data['u'] = u2_in
        # self.data['T'] = np.where(self.iron_mask, EOS.T2_iron_EOS(self.data['u'], self.data['rho']), EOS.T2_fst_EOS(self.data['u'], self.data['rho']))
        # self.data['P'] = np.where(self.iron_mask, EOS.P_iron_EOS(self.data['rho'], self.data['T']), EOS.P_fst_EOS(self.data['rho'], self.data['T']))
        # self.data['S'] = np.where(self.iron_mask, EOS.S_iron_EOS(self.data['rho'], self.data['T']), EOS.S_fst_EOS(self.data['rho'], self.data['T']))
        # self.calculate_EOS()

        if self.verbose:
            print(f'Cooling by {du_in / u_avg_in:.3%} over {dt / (3600 * 24):.2f} days')
            print(f'Energy loss inner region: {dE_in:.2e} ({du_in / u_avg_in:.3%})')
        # print(f'Energy loss outer region: {dE_out:.2e} ({du_out / u_avg_out:.3%})')

    def initial_cool_v2(self, max_time):

        tau = self.data['alpha_v'] * self.data['dr']
        emissivity = 1 - np.exp(-tau) + tau * exp1(tau)
        L = sigma * self.data['T'] ** 4 * self.data['A_r+'] * emissivity
        t_cool = (self.data['E'] / L) * (R_earth / self.data['dr'])

        min_cooling_time = np.nanmin(t_cool)
        dt = min_cooling_time / 4

        if dt > max_time:
            if self.verbose:
                print('Max initial cooling time exceeded')
            return False

        if self.verbose:
            print(f'Initial cool for {dt:.1e} seconds')

        u1, rho, T1 = self.data['u'], self.data['rho'], np.array(self.data['T'])
        k = np.minimum(dt / t_cool, 0.99)
        du = k * u1
        u2 = u1 - du
        T2 = EOS.T2_fst_EOS(u2, rho)

        self.data['T'] = T2
        self.data['P'] = EOS.P_fst_EOS(rho, T2)
        self.data['s'] = EOS.S_fst_EOS(rho, T2)
        self.calculate_EOS()

        # self.data['u'] = u2
        # self.data['T'] = np.where(self.iron_mask, EOS.T2_iron_EOS(self.data['u'], self.data['rho']),
        #                           EOS.T2_fst_EOS(self.data['u'], self.data['rho']))
        # self.data['P'] = np.where(self.iron_mask, EOS.P_iron_EOS(self.data['rho'], self.data['T']),
        #                           EOS.P_fst_EOS(self.data['rho'], self.data['T']))
        # self.data['S'] = np.where(self.iron_mask, EOS.S_iron_EOS(self.data['rho'], self.data['T']),
        #                           EOS.S_fst_EOS(self.data['rho'], self.data['T']))
        # self.calculate_EOS()

        return True

    def long_term_evolution(self, n, dt):

        phot.verbose = False

        total_mass_loss = 0
        L = [self.luminosity]
        t = [0]
        dt, n = 1e7, 400

        print(f'Cooling for {n * dt:.1e} seconds ({n * dt / yr:.2f} years):')

        for i in tqdm(range(n)):
            self.cool_step(dt)
            total_mass_loss += self.remove_droplets()
            self.get_photosphere()

            t.append(i * dt)
            L.append(self.luminosity)

        t = np.array(t) / yr
        L = np.array(L) / L_sun

        phot.verbose = True

        return t, L


if __name__ == '__main__':

    snap = snapshot('snapshots/basic_twin/snapshot_0411.hdf5')
    phot = photosphere(snap, resolution=500, n_theta=40, max_size=100*Rearth)
    phot.get_photosphere()
    phot.plot('T', plot_photosphere=True, round_to=1000, log=False, val_max=8000)
    phot.initial_cool_v2(1e5)
    phot.plot('T', plot_photosphere=True, round_to=1000, log=False, val_max=8000)

    while phot.initial_cool_v2(1e5):
        pass

    phot.remove_droplets()
    phot.get_photosphere()
    phot.plot('T', plot_photosphere=True, round_to=1000, log=False, val_max=8000)

    t, L = phot.long_term_evolution(200, 1e7)

    plt.plot(t, L)
    plt.show()


# new sims that work: 0, 1, 3, 4, 6, 8
# 8, 9 requires cooling
# 4, 6 has some weird bits
# 5 also (not really a synestia)
