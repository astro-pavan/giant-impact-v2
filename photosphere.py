# type: ignore

# extracts data from a snapshot and analyses it, producing a 1D photosphere model
# all values are in SI units unless otherwise specified

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from tqdm import tqdm

from swiftsimio.visualisation import slice_gas, project_gas
from unyt import Rearth
import woma

woma.load_eos_tables()

from snapshot_analysis import snapshot, gas_slice
import EOS as fst
import EOS2

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

class photosphere:

    def __init__(self, filename, pressure_floor=1e9, remove_droplets=True, orbital_period=10 * day):

        self.filename = filename
        self.snapshot = snapshot(filename)
        # Ensure swiftsimio treats snapshot coordinates as comoving (even if
        # there is no cosmology information), so that slicing routines do not
        # attempt invalid conversions.
        try:
            self.snapshot.data.gas.coordinates.comoving = True
            self.snapshot.data.gas.coordinates.valid_transform = True
            self.snapshot.data.gas.metadata.boxsize.comoving = True
            self.snapshot.data.gas.metadata.boxsize.valid_transform = True
        except Exception:
            pass

        self.droplet_removal = remove_droplets
        self.pressure_floor = pressure_floor
        self.orbital_period = orbital_period

        max_size_mks = ((G * self.snapshot.total_mass * (orbital_period ** 2)) / (12 * np.pi * np.pi)) ** (1 / 3)
        # max_size = (max_size_mks / 6371000) * Rearth
        #print(f'Hill radius = {max_size}')

        resolution = 400
        sample_size = np.minimum(10, max_size_mks / R_earth) * Rearth
        max_size = max_size_mks

        # calculate the center of the snapshot and set the limits for the slice
        center = self.snapshot.center_of_mass
        limits = [
            center[0] - sample_size,
            center[0] + sample_size,
            center[1] - sample_size,
            center[1] + sample_size,
        ]

        # ensure the region has the same comoving/physical metadata as the snapshot
        from swiftsimio.objects import cosmo_array

        region = cosmo_array(
            limits,
            comoving=self.snapshot.data.gas.coordinates.comoving,
            cosmo_factor=self.snapshot.data.gas.coordinates.cosmo_factor,
        )

        n_theta = 100
        r_range = np.array(np.linspace(0, sample_size.value * 0.95, num=resolution // 2 + 1)[1:]) * Rearth
        theta_range = np.arange(n_theta+1) * (pi / n_theta)
        r_2d, theta_2d = np.meshgrid(r_range, theta_range)

        pixel_size = sample_size.value / (resolution // 2)

        i_x = np.int32((r_2d.value * np.cos(theta_2d) / pixel_size) + (resolution / 2))
        i_y = np.int32((r_2d.value * np.sin(theta_2d) / pixel_size) + (resolution / 2))

        indexes = i_y, i_x

        # loads density slice
        mass_slice = slice_gas(
            self.snapshot.data,
            z_slice=center[2],
            resolution=resolution,
            project="masses",
            region=region,
            parallel=True,
        )

        # function that loads the slice of each property
        def get_slice(parameter):

            mass_weighted_slice = slice_gas(
                self.snapshot.data,
                z_slice=center[2],
                resolution=resolution,
                project=f'{parameter}_mass_weighted',
                region=region,
                parallel=True,
            )

            property_slice = mass_weighted_slice / mass_slice

            return property_slice[tuple(indexes)]

        mass_slice.convert_to_mks()

        # --- diagnostic: save slice image so we can verify the body is captured ---
        import os
        _ss = float(sample_size.value)           # extent in R_earth
        _slice_arr = np.array(mass_slice)
        _sim_label = os.path.basename(os.path.dirname(self.filename))
        _snap_label = os.path.splitext(os.path.basename(self.filename))[0]
        _fig, _ax = plt.subplots(figsize=(5, 4.5))
        _im = _ax.imshow(
            np.log10(np.maximum(_slice_arr, 1e-30)),
            origin='lower',
            extent=[-_ss, _ss, -_ss, _ss],
            cmap='viridis',
        )
        plt.colorbar(_im, ax=_ax, label=r'$\log_{10}$(mass slice) [MKS]')
        _ax.set_xlabel(r'$x$ ($R_\oplus$)')
        _ax.set_ylabel(r'$y$ ($R_\oplus$)')
        _ax.set_title(f'{_sim_label}\n{_snap_label}', fontsize=7)
        # circle showing the outer edge of the radial sampling region
        _theta_circ = np.linspace(0, 2 * np.pi, 300)
        _ax.plot(_ss * 0.95 * np.cos(_theta_circ),
                 _ss * 0.95 * np.sin(_theta_circ),
                 'w--', linewidth=0.8, label='sample edge')
        _ax.legend(fontsize=7, loc='upper right')
        _fig.tight_layout()
        os.makedirs('figures/diagnostics', exist_ok=True)
        _fig.savefig(f'figures/diagnostics/slice_{_sim_label}_{_snap_label}.png',
                     dpi=120, bbox_inches='tight')
        plt.close(_fig)
        # -------------------------------------------------------------------------

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

        # mass_projection = project_gas(self.snapshot.data,
        #                        resolution=5096,
        #                        project="masses",
        #                        parallel=True,
        #                        region=limits
        #                        )
        
        # mass_weighted_projection = project_gas(
        #         self.snapshot.data,
        #         resolution=5096,
        #         project=f'entropy_mass_weighted',
        #         parallel=True, 
        #         region=limits
        #     )
        
        # entropy_projection = mass_weighted_projection / mass_projection

        # plt.imshow(mass_projection)
        # plt.savefig('mass_projection.png')
        # plt.close()
        
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

        # self.aspect_ratio = self.snapshot.HD_limit_z.value / self.snapshot.HD_limit_R.value # ellipse correction factor
        self.aspect_ratio = 1

        self.dV = (4 * pi * self.r ** 2 * self.dr) * self.aspect_ratio

        self.extrapolate_r = self.snapshot.HD_limit_R.value  # density floor limit in meters

        if self.extrapolate_r < max_size:

            self.extrapolation_index = np.argmax(self.r > self.extrapolate_r)
            self.s_extrapolation_value = self.s[self.extrapolation_index - 1]

            self.s = np.where(self.r < self.extrapolate_r, self.s, self.s_extrapolation_value)
            self.s_interpolation = CubicSpline(self.r, self.s)

        self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400))
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u

        self.omega = self.snapshot.best_fit_rotation_curve_mks(self.r)
        self.omega_keplerian = np.sqrt((G * self.snapshot.total_mass) / (self.r ** 3))

        self.R_phot, self.T_phot, self.L_phot, self.P_phot = 0, 0, 0, 0

        if self.extrapolate_r < max_size:
            self.solve_dPdr()

        self.remove_droplets()
        self.calculate_luminosity()
        self.save_diagnostic_profile()

    def solve_dPdr(self):

        rho_func, T_func = EOS2.make_isentrope(self.s_extrapolation_value)
        
        def dPdr(r, P):
            
            s = self.s_extrapolation_value
            omega = self.snapshot.best_fit_rotation_curve_mks(r)
            M = self.snapshot.total_mass

            lP  = np.log(np.maximum(P, 1e-300))
            rho = rho_func(lP)
            gravity = - G * M / (r ** 2)
            centrifugal = r * omega ** 2

            result = rho * (gravity + centrifugal)
            result = np.where(rho > 1e-8, result, 0)

            return result

        solution = solve_ivp(
            dPdr,
            (self.extrapolate_r, self.r[-1]),
            [self.P[self.extrapolation_index - 1]],
            t_eval=self.r[self.extrapolation_index:]
            )

        P_sol   = np.maximum(solution.y[0], 0)
        lP_sol  = np.log(np.maximum(P_sol, 1e-300))
        rho_sol = np.maximum(rho_func(lP_sol), 0)
        T_sol   = np.maximum(T_func(lP_sol),   0)
        u_sol = woma.A1_u_rho_T(rho_sol, T_sol, np.full_like(rho_sol, 400))

        self.P[self.extrapolation_index:] = P_sol

        self.rho[self.extrapolation_index:] = rho_sol
        self.T[self.extrapolation_index:] = T_sol
        self.u[self.extrapolation_index:] = u_sol

        # self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
        # self.T = fst.T1_EOS(self.s, self.P)
        # self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400)) 

        # self.remove_nans()

        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u 
        
    def apply_EOS(self):
        
        self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
        self.T = fst.T1_EOS(self.s, self.P)
        # self.u = fst.u_EOS(self.s, self.P)

        self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400)) 
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u

    def EOS_from_Ps(self):
        
        self.dm = self.rho * self.dV
        self.dE = self.dm * self.u

    def EOS_from_rho_u(self):

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

    def remove_droplets(self, max_infall_time=1e4, override=False):

        if self.droplet_removal or override:

            phase = EOS2.phase(self.P, self.s)
            
            D0 = 1e-3
            CD = 0.5

            rho_droplet = fst.rho_liquid(self.P)
            rho_vapour = fst.rho_vapor(self.rho, self.s, self.P)

            v_relative = np.abs(self.r * (self.omega_keplerian - self.omega))
            v_orbit = self.r * self.omega_keplerian
            t_infall = (2 * rho_droplet * D0 * v_orbit) / (rho_vapour * CD * (v_relative ** 2))

            condensation_mask = phase == 2
            remove_mask = condensation_mask & (t_infall < max_infall_time)

            # Phase-1 (fully condensed) cells in the outer low-pressure atmosphere
            # accumulate high opacity (alpha=1e14) and stay optically thick, causing
            # cool_step to keep draining them until they cycle back into phase 2 and
            # the photosphere re-emerges.  Treat them as condensate that has rained out.
            phase1_outer = (phase == 1) & (self.P < self.pressure_floor)
            remove_mask = remove_mask | phase1_outer

            new_rho, new_T, new_s, new_u = EOS2.vapor_curve(self.P)
            new_rho = np.maximum(new_rho, 0)
            new_T   = np.maximum(new_T,   0)
            new_u   = np.maximum(new_u,   0)

            self.rho = np.where(remove_mask, new_rho, self.rho)
            self.T = np.where(remove_mask, new_T, self.T)
            self.s = np.where(remove_mask, new_s, self.s)
            self.u = np.where(remove_mask, new_u, self.u)

            # new_s = fst.condensation_S(self.s, self.P)
            # self.s = np.where(remove_mask, new_s, self.s)
            # self.rho = np.nan_to_num(fst.rho_EOS(self.s, self.P))
            # self.T = fst.T1_EOS(self.s, self.P)
            # self.u = woma.A1_u_rho_T(self.rho, self.T, np.full_like(self.rho, 400))

            # self.remove_nans()
            self.dm = self.rho * self.dV
            self.dE = self.dm * self.u

    def calculate_luminosity(self):
        
        self.alpha = fst.alpha(self.rho, self.T, self.P, self.s, D0=0.001)
        self.alpha_v = fst.alpha_v(self.rho, self.T)

        d_tau = np.maximum(self.alpha, 0) * self.dr
        self.tau = np.flip(np.cumsum(np.flip(d_tau)))

        tau_below_1 = self.tau < 1
        if np.any(tau_below_1):
            photosphere_index = np.argmax(tau_below_1)
            # If this index falls in the un-extrapolated extension region (T=0),
            # fall back to the last sampled cell with a valid temperature.
            if self.T[photosphere_index] == 0:
                valid_T_idx = np.where(self.T > 0)[0]
                if len(valid_T_idx) > 0:
                    photosphere_index = valid_T_idx[-1]
        else:
            # Entire domain is optically thick; photosphere lies at or beyond the
            # Hill sphere boundary.  Use the outermost cell as the best estimate.
            photosphere_index = len(self.tau) - 1

        self.R_phot = self.r[photosphere_index]
        self.T_phot = self.T[photosphere_index]
        self.P_phot = self.P[photosphere_index - 1]

        eccentricity = np.sqrt(1 - self.aspect_ratio ** 2)
        ellipse_correction_factor = 0.5 * (1 + np.arctanh(eccentricity) * (1 - eccentricity ** 2) / eccentricity)
        A_phot = 4 * pi * self.R_phot ** 2 

        self.L_phot = A_phot * sigma * (self.T_phot ** 4)

        # print(f'Luminosity : {self.L_phot / 3.8e26:.2e} L_sun')

    def cool_step(self, dt):

        inside_photosphere_mask = self.tau > 1
        pressure_mask = self.P < self.pressure_floor

        E_inner_region = np.sum(self.dE[inside_photosphere_mask & pressure_mask])
        m_inner_region = np.sum(self.dm[inside_photosphere_mask & pressure_mask])

        if m_inner_region <= 0:
            # print(self.P_phot)
            self.plot('error')
            return True

        if E_inner_region <= 0:
            # Outer atmosphere energy exhausted (e.g. all condensate removed and
            # remaining cells have u=0).  Nothing left to drain — end cooling.
            return True

        E_lost = self.L_phot * dt
        u_avg = E_inner_region / m_inner_region
        u_avg_lost = E_lost / m_inner_region
        loss_factor = (1 - u_avg_lost / u_avg) if m_inner_region > 1 else 1
        loss_factor = max(loss_factor, 0)  # clamp: a single step cannot remove more than all energy

        assert loss_factor <= 1

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

        # if np.any(self.P < 1e-6):
        #     print(self.u)
        #     plt.plot(self.r, self.u)
        #     plt.plot(self.r, old_u)
        #     plt.xscale('log')
        #     plt.yscale('log')
        #     plt.savefig('Pu.png')
        #     plt.close()
        #     self.plot('error')

        # assert np.all(self.P > 1e-6)

        if np.sum(self.dE[inside_photosphere_mask & pressure_mask]) <= 0:
            self.plot('error')
            _fig_rP, _ax_rP = plt.subplots()
            _ax_rP.loglog(self.r, self.P)
            _fig_rP.savefig('rP.png')
            plt.close(_fig_rP)
            _fig_ru, _ax_ru = plt.subplots()
            _ax_ru.loglog(self.r, self.u)
            _fig_ru.savefig('ru.png')
            plt.close(_fig_ru)

        return False

    def cool(self, max_time, n=100, t_min=None, max_dt=None):

        if t_min is None:
            t_min = max_time / n / 100  # 100× finer than uniform at the start

        # Build a time grid that always reaches max_time.
        # If max_dt is set: log-spaced up to where the step size would first reach max_dt,
        # then uniform max_dt steps to max_time.  This guarantees no step exceeds max_dt.
        if max_dt is not None:
            # Step size in a geomspace grows as t * (exp(log_ratio) - 1).
            # Find the crossover time where this equals max_dt.
            log_ratio = np.log(max_time / t_min) / n
            t_crossover = max_dt / np.expm1(log_ratio)
            t_crossover = min(t_crossover, max_time)
            n_log = max(2, round(np.log(t_crossover / t_min) / log_ratio))
            t_log = np.geomspace(t_min, t_crossover, n_log)
            if t_crossover < max_time:
                n_lin = int(np.ceil((max_time - t_crossover) / max_dt))
                t_lin = t_crossover + np.arange(1, n_lin + 1) * max_dt
                t_lin[-1] = max_time
                t_edges = np.concatenate([[0], t_log, t_lin])
            else:
                t_edges = np.concatenate([[0], t_log])
        else:
            t_edges = np.concatenate([[0], np.geomspace(t_min, max_time, n)])

        t_current = 0

        t, L, R, T = [t_current], [self.L_phot], [self.R_phot], [self.T_phot]

        L_init = self.L_phot

        for i in range(len(t_edges) - 1):

            if self.L_phot < L_init / 100:
                break

            dt = t_edges[i + 1] - t_edges[i]
            t_current += dt
            end = self.cool_step(dt)
            self.remove_droplets()
            self.calculate_luminosity()

            t.append(t_current)
            L.append(self.L_phot)
            R.append(self.R_phot)
            T.append(self.T_phot)

            if end:
                break

        t, L, R, T = np.array(t), np.array(L), np.array(R), np.array(T)

        half_mask = (L / L[0]) <= 0.5
        tenth_mask = (L / L[0]) <= 0.1
        t_half  = t[np.argmax(half_mask)]  if np.any(half_mask)  else t[-1]
        t_tenth = t[np.argmax(tenth_mask)] if np.any(tenth_mask) else t[-1]

        self._save_cooling_diagnostic(t, L, R, T, t_half)

        return t, L, R, T, t_half, t_tenth

    def _save_cooling_diagnostic(self, t, L, R, T, t_half):
        import os

        sim_label   = os.path.basename(os.path.dirname(self.filename))
        snap_label  = os.path.splitext(os.path.basename(self.filename))[0]
        period_days = self.orbital_period / day

        t_days = t / day
        A      = 4 * pi * R ** 2          # photosphere area  [m²]
        A_Re2  = A / (4 * pi * R_earth**2) # in units of R_earth²

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].semilogy(t_days, L / L_sun)
        axes[0].axvline(t_half / day, color='red', ls='--', lw=0.9, label=f'$t_{{1/2}}$ = {t_half/day:.1f} d')
        axes[0].set_ylabel(r'$L$ [$L_\odot$]')
        axes[0].legend(fontsize=7)

        axes[1].plot(t_days, T)
        axes[1].set_ylabel(r'$T_\mathrm{phot}$ [K]')

        axes[2].semilogy(t_days, R / R_earth)
        axes[2].set_ylabel(r'$R_\mathrm{phot}$ [$R_\oplus$]')
        axes[2].set_xlabel('Time [days]')

        for ax in axes:
            ax.grid(True, which='both', ls='--', lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=7)

        fig.suptitle(f'{sim_label}   |   {snap_label}', fontsize=7, y=1.002)
        fig.tight_layout()

        os.makedirs('figures/diagnostics', exist_ok=True)
        fig.savefig(f'figures/diagnostics/cooling_{sim_label}_{snap_label}_P{period_days:.0f}d.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    
    def save_diagnostic_profile(self):
        import os

        sim_label  = os.path.basename(os.path.dirname(self.filename))
        snap_label = os.path.splitext(os.path.basename(self.filename))[0]

        r_re = self.r / R_earth

        quantities = [self.P, self.T, self.s, self.u, self.alpha, self.tau]
        labels = [
            r'$P$ [Pa]',
            r'$T$ [K]',
            r'$s$ [J K$^{-1}$ kg$^{-1}$]',
            r'$u$ [J kg$^{-1}$]',
            r'$\alpha$ [m$^{-1}$]',
            r'$\tau$',
        ]

        fig, axes = plt.subplots(6, 1, figsize=(8, 14), sharex=True)

        for ax, q, label in zip(axes, quantities, labels):
            safe = np.where(np.isfinite(q) & (q > 0), q, np.nan)
            ax.semilogy(r_re, safe, linewidth=0.9)
            ax.set_ylabel(label, fontsize=8)
            ax.axvline(self.R_phot / R_earth, color='red',  ls='--', lw=0.9, label='$R_\mathrm{phot}$')
            if hasattr(self, 'extrapolate_r'):
                ax.axvline(self.extrapolate_r / R_earth, color='0.5', ls=':', lw=0.8, label='extrap. boundary')
            ax.grid(True, which='both', ls='--', lw=0.3, alpha=0.5)
            ax.tick_params(labelsize=7)

        # horizontal tau = 1 reference line
        axes[-1].axhline(1, color='orange', ls='--', lw=0.9, label=r'$\tau = 1$')
        axes[-1].set_xlabel(r'$R$ [$R_\oplus$]', fontsize=9)

        axes[0].legend(fontsize=6, loc='upper right', ncol=3)

        fig.suptitle(f'{sim_label}   |   {snap_label}', fontsize=7, y=1.002)
        fig.tight_layout()

        os.makedirs('figures/diagnostics', exist_ok=True)
        fig.savefig(f'figures/diagnostics/profile_{sim_label}_{snap_label}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)

    def plot(self, filename):
        fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        axs = axs.flatten()

        quantities = [self.rho, self.T, self.P, self.s, self.u, self.tau]
        labels = [r'$\rho$ [kg/m$^3$]', 'T [K]', 'P [Pa]', 's [J/K/kg]', 'u [J/kg]', r'$\tau$']
        titles = [r'Density $\rho$', 'Temperature T', 'Pressure P', 'Entropy s', 'Internal Energy u', 'Optical Depth τ']
        ylim = [(1e-10, 1e4), (1000, 10000), (1e-3, 1e11), (2000, 12000), (1e6, 5e7), (1e-7, 1e20)]

        for i, (ax, y, label, title) in enumerate(zip(axs, quantities, labels, titles)):
            y = np.nan_to_num(y, posinf=0, neginf=0)
            assert np.all(np.isfinite(y))
            assert np.all(np.isfinite(self.r))
            ax.plot(self.r / R_earth, y)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylabel(label)
            ax.set_xlim([0, 100])
            ax.axvline(self.extrapolate_r / R_earth)
            ax.axvline(self.R_phot / R_earth)
            # ax.set_ylim(ylim[i])
            ax.grid(True, which='both', ls='--', lw=0.5)

            plt.savefig(f'{filename}.png')

        for ax in axs[4:]:
            ax.set_xlabel('r [R_Earth]')

        plt.tight_layout()

        plt.savefig(f'{filename}.png')
        plt.close()


if __name__ == "__main__":

    from test import get_filename

    # 21 may be a bad simulation (there appear to be 3 remnants)
    # 24 and 25 have strange cooling curves

    p1 = photosphere(get_filename(0, 4), pressure_floor=1e9, orbital_period=1 * day) # 21*, 24*, 25*
    p1.plot('profile')
    t, L, R, T, t_half, t_tenth = p1.cool(20 * yr, n=10000)
    p1.plot('profile2')

    plt.plot(t / yr, L / L_sun)
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('L')
    plt.savefig('cooling.png')