# type: ignore

# extracts data from a snapshot and analyses it, producing a 1D photosphere model
# all values are in SI units unless otherwise specified

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from tqdm import tqdm

from swiftsimio.visualisation import slice_gas
from unyt import Rearth
import woma

from snapshot_analysis import snapshot
import EOS as fst

sigma = 5.670374419e-8 # stefan-boltzmann constant
L_sun = 3.828e26 # solar luminosity in watts
R_earth = 6371000 # radius of Earth in meters
M_earth = 5.972e24 # mass of Earth in kg
G = 6.67430e-11 # gravitational constant in m^3 kg^-1 s^-2
day = 3600 * 24
yr = 365.25 * day
silicate_latent_heat_v = 3e7
photosphere_depth = 2/3
pi = np.pi
cos = lambda theta: np.cos(theta)
sin = lambda theta: np.sin(theta)

woma.load_eos_tables(['ANEOS_forsterite'])

class photosphere:

    def __init__(self, filename):

        self.filename = filename
        self.snapshot = snapshot(filename)

        resolution = 200
        sample_size = 10 * Rearth
        max_size = 100 * 6371000

        # calculate the center of the snapshot and set the limits for the slice
        center = self.snapshot.center_of_mass
        limits = [center[0] - sample_size, center[0] + sample_size, center[1] - sample_size, center[1] + sample_size]

        n_theta = 100
        r_range = np.array(np.linspace(0, sample_size.value * 0.95, num=resolution // 2 + 1)[1:]) * Rearth
        theta_range = np.arange(n_theta+1) * (pi / n_theta)
        r_2d, theta_2d = np.meshgrid(r_range, theta_range)

        pixel_size = sample_size.value / (resolution // 2)

        i_x = np.int32((r_2d.value * np.cos(theta_2d) / pixel_size) + (resolution / 2))
        i_y = np.int32((r_2d.value * np.sin(theta_2d) / pixel_size) + (resolution / 2))

        indexes = i_y, i_x

        # loads density slice
        mass_slice = slice_gas(self.snapshot.data,
                               z_slice=center[2],
                               resolution=resolution,
                               project="masses",
                               region=limits,
                               parallel=True
                               )

        # function that loads the slice of each property
        def get_slice(parameter):

            mass_weighted_slice = slice_gas(
                self.snapshot.data,
                z_slice=center[2],
                resolution=resolution,
                project=f'{parameter}_mass_weighted',
                region=limits,
                parallel=True
            )

            property_slice = mass_weighted_slice / mass_slice

            return property_slice[tuple(indexes)]
        
        mass_slice.convert_to_mks()
        
        # loading slices of each property
        densities = mass_slice[tuple(indexes)]
        temperatures = get_slice('temperatures')
        pressures, entropies = get_slice('pressures'), get_slice('entropy')

        # convert data to MKS
        temperatures.convert_to_mks()
        pressures.convert_to_mks()
        entropies.convert_to_mks()
        r_range.convert_to_mks()

        self.rho = np.array(np.mean(densities, axis=0))
        self.T = np.array(np.mean(temperatures, axis=0))
        self.P = np.array(np.mean(pressures, axis=0))
        self.s = np.array(np.mean(entropies, axis=0))
        self.r = np.array(r_range)

        r_extension = np.linspace(self.r[-1], max_size, num=resolution // 2)

        self.r = np.concatenate((self.r, r_extension[1:]))
        self.rho = np.concatenate((self.rho, np.zeros_like(r_extension[1:])))
        self.T = np.concatenate((self.T, np.zeros_like(r_extension[1:])))
        self.P = np.concatenate((self.P, np.zeros_like(r_extension[1:])))
        self.s = np.concatenate((self.s, np.zeros_like(r_extension[1:])))
        self.u = np.zeros_like(self.r)
        self.dE = np.zeros_like(self.r)
        self.alpha = np.zeros_like(self.r)
        self.alpha_v = np.zeros_like(self.r)
        self.tau = np.zeros_like(self.r)

        dr = np.diff(self.r)
        self.dr = np.concatenate([dr, [dr[-1]]])

        self.dV = 4 * pi * self.r ** 2 * self.dr

        self.extrapolate_r = self.snapshot.HD_limit_R.value  # density floor limit in meters
        self.extrapolation_index = np.argmax(self.r > self.extrapolate_r)
        s_extrapolation_value = self.s[self.extrapolation_index - 1]

        self.s = np.where(self.r < self.extrapolate_r, self.s, s_extrapolation_value)
        self.s_interpolation = CubicSpline(self.r, self.s)

        self.omega = self.snapshot.best_fit_rotation_curve_mks(self.r)
        self.omega_keplerian = np.sqrt((G * self.snapshot.total_mass) / (self.r ** 3))

        self.R_phot, self.T_phot, self.L_phot = 0, 0, 0

        print('Loaded data')
        print('')

        self.solve_dPdr()
        # self.apply_EOS()
        self.remove_droplets()
        self.calculate_luminosity()

    def solve_dPdr(self):
        
        def dPdr(r, P):
            
            s = self.s_interpolation(r)
            omega = self.snapshot.best_fit_rotation_curve_mks(r)
            M = self.snapshot.total_mass

            rho = fst.rho_EOS(s, P)
            gravity = - G * M / (r ** 2)
            centrifugal = r * omega ** 2

            result = rho * (gravity + centrifugal)
            
            assert rho >= 0

            return result
        
        solution = solve_ivp(
            dPdr,
            (self.extrapolate_r, self.r[-1]),
            [self.P[self.extrapolation_index - 1]],
            t_eval=self.r[self.extrapolation_index:]
            )
        
        self.P[self.extrapolation_index:] = solution.y[0]
        self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
        self.T = fst.T1_EOS(self.s, self.P)
        self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400)) 
        self.remove_nans()
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u   
        
    def apply_EOS(self):
        
        self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
        self.T = fst.T1_EOS(self.s, self.P)
        # self.u = fst.u_EOS(self.s, self.P)

        self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400)) 
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u  

    def remove_nans(self):
        self.rho = np.nan_to_num(self.rho)
        self.T = np.nan_to_num(self.T)
        self.P = np.nan_to_num(self.P)
        self.s = np.nan_to_num(self.s)
        self.u = np.nan_to_num(self.u)
        # self.dE = np.nan_to_num(self.dE)
        # self.alpha = np.nan_to_num(self.alpha)
        # self.alpha_v = np.nan_to_num(self.alpha_v)
        # self.tau = np.nan_to_num(self.tau)

    def remove_droplets(self, max_infall_time=1e4):

        phase = fst.phase(self.s, self.P)
        
        D0 = 1e-3
        CD = 0.5

        rho_droplet = fst.rho_liquid(self.P)
        rho_vapour = fst.rho_vapor(self.rho, self.s, self.P)

        v_relative = np.abs(self.r * (self.omega_keplerian - self.omega))
        v_orbit = self.r * self.omega_keplerian
        t_infall = (2 * rho_droplet * D0 * v_orbit) / (rho_vapour * CD * (v_relative ** 2))

        condensation_mask = phase == 2
        remove_mask = condensation_mask & (t_infall < max_infall_time)

        new_s = fst.condensation_S(self.s, self.P)
        self.s = np.where(remove_mask, new_s, self.s)
        self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
        self.T = fst.T1_EOS(self.s, self.P)
        self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400))
        self.remove_nans()
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u
        # self.apply_EOS()


    def calculate_luminosity(self):
        
        self.alpha = fst.alpha(self.rho, self.T, self.P, self.s, D0=0.001)
        self.alpha_v = fst.alpha_v(self.rho, self.T)

        d_tau = self.alpha_v * self.dr
        self.tau = np.flip(np.cumsum(np.flip(d_tau)))

        photosphere_index = np.argmax(self.tau < 1)

        self.R_phot = self.r[photosphere_index]
        self.T_phot = self.T[photosphere_index]
        self.L_phot = 4 * pi * (self.R_phot ** 2) * sigma * (self.T_phot ** 4)

        print(f'Luminosity : {self.L_phot / 3.8e26:.2e} L_sun')

    def cool_step(self, dt):
        
        # self.plot('a')

        inside_photosphere_mask = self.tau > 1
        pressure_mask = self.P < 1e10

        E_inner_region = np.sum(self.dE[inside_photosphere_mask & pressure_mask])
        m_inner_region = np.sum(self.dm[inside_photosphere_mask & pressure_mask])

        assert m_inner_region > 0
        assert E_inner_region > 0

        E_lost = self.L_phot * dt
        u_avg = E_inner_region / m_inner_region
        u_avg_lost = E_lost / m_inner_region
        loss_factor = (1 - u_avg_lost / u_avg) if m_inner_region > 1 else 1

        print(f'Energy : {E_inner_region:.2e} J')
        print(f'Cooling time : {E_inner_region / self.L_phot:.2e} s')
        print(f'dt : {dt:.2e}')
        print(f'Loss factor : {loss_factor}')
        print('')

        assert loss_factor <= 1
        assert E_inner_region > 0

        self.u = np.where(pressure_mask, self.u * loss_factor, self.u)
        # self.T = np.nan_to_num(fst.T2_EOS(self.u, self.rho))
        self.T = woma.A1_T_rho_u(self.rho, self.u, np.full_like(self.rho, 400))
        # self.P = fst.P_EOS(self.rho, self.T)
        # self.s = fst.S_EOS(self.rho, self.T)
        self.P = woma.A1_P_rho_u(self.rho, self.u, np.full_like(self.rho, 400))
        self.s = woma.A1_s_rho_u(self.rho, self.u, np.full_like(self.rho, 400))
        self.remove_nans()
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u

        # self.plot('b')

        # assert np.all(self.P > 1e-3)

        assert np.sum(self.dE[inside_photosphere_mask & pressure_mask]) > 0

    def cool(self, max_time, n=100):

        dt = max_time / n
        t_current = 0

        t, L, R, T = [t_current], [self.L_phot], [self.R_phot], [self.T_phot]

        for i in range(n):

            t_current += dt
            self.cool_step(dt)
            self.remove_droplets(dt)
            self.calculate_luminosity()

            t.append(t_current)
            L.append(self.L_phot)
            R.append(self.R_phot)
            T.append(self.T_phot)

        t, L, R, T = np.array(t), np.array(L), np.array(R), np.array(T)

        i_half = np.argmin((L / L[0]) > 0.5)
        i_tenth = np.argmin((L / L[0]) > 0.1)
        t_half, t_tenth = t[i_half], t[i_tenth]

        return t, L, R, T, t_half, t_tenth
    
    def plot(self, filename):
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        axs = axs.flatten()

        quantities = [self.rho, self.T, self.P, self.s, self.u, self.tau]
        labels = [r'$\rho$ [kg/m$^3$]', 'T [K]', 'P [Pa]', 's [J/K/kg]', 'u [J/kg]', r'$\tau$']
        titles = [r'Density $\rho$', 'Temperature T', 'Pressure P', 'Entropy s', 'Internal Energy u', 'Optical Depth τ']
        ylim = [(1e-3, 1e4), (1000, 8000), (1e3, 1e11), (2000, 10000), (1e7, 3e7), (1e-7, 1e20)]

        for i, (ax, y, label, title) in enumerate(zip(axs, quantities, labels, titles)):
            y = np.nan_to_num(y, posinf=0, neginf=0)
            assert np.all(np.isfinite(y))
            assert np.all(np.isfinite(self.r))
            ax.plot(self.r / R_earth, y)
            ax.set_yscale('log')
            ax.set_ylabel(label)
            ax.set_xlim([0, 25])
            ax.set_ylim(ylim[i])
            ax.grid(True, which='both', ls='--', lw=0.5)

            plt.savefig(f'{filename}.png')

        for ax in axs[4:]:
            ax.set_xlabel('r [R_Earth]')

        plt.tight_layout()
        plt.savefig(f'{filename}.png')
        plt.close()


if __name__ == "__main__":

    p1 = photosphere("snapshot_0240.hdf5")
    t, L, R, T, t_half, t_tenth = p1.cool(5 * yr, n=2000)

    plt.plot(t / yr, L / L_sun)
    plt.xlabel('t')
    plt.ylabel('L')
    plt.savefig('cooling.png')