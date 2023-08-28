# creates the initial conditions

import woma
import numpy as np

import os
import shutil

working_directory = '../'

R_earth = 6.371e6  # m
M_earth = 5.9724e24  # kg m^-3


# creates initial condition hdf5 file
def generate_initial_conditions(particle_count, target_mass, mass_ratio, impact_velocity, impact_parameter,
                                spin_period=0, velocity_unit='v_esc', start_time=1800):

    name = f'impact_p{particle_count:.1e}_M{target_mass:.1f}_ratio{mass_ratio:.2f}_v{impact_velocity:.2f}_b{impact_parameter:.2f}_spin{spin_period:.1f}'

    create_sim_directory(name)
    create_parameter_file(name, sim_time=144000, snapshot_frequency=600)
    create_slurm_script(name)
    create_restart(name)

    total_mass = target_mass * (1 + mass_ratio)
    impactor_mass = target_mass * mass_ratio

    # Build an approximation to the planet
    print("BUILDING TARGET PLANET")
    try:

        target = woma.Planet(
            name="target_planet",
            A1_mat_layer=["ANEOS_iron", "ANEOS_forsterite"],
            # check the name of different material and EoS at: https://github.com/srbonilla/WoMa/blob/master/woma/misc/glob_vars.py
            A1_T_rho_type=["entropy=1750", "entropy=3027"],
            # Here I use fixed entropy for each layer, other option: "adiabatic","power=<float>"
            P_s=1e9,  # Surface pressure
            T_s=1500,  # Surface temperature
            M=target_mass * M_earth,
            R=R_earth * (target_mass ** 0.33)
        )

        target.gen_prof_L2_find_R1_given_M_R()

    except:

        print('Core/mantle ratio not viable, trying again')
        target.R = R_earth * (target_mass ** 0.2)
        target.gen_prof_L2_find_R1_given_M_R()

    print("BUILDING IMPACTING PLANET")

    try:

        impactor = woma.Planet(
            name="impacting_planet",
            A1_mat_layer=["ANEOS_iron", "ANEOS_forsterite"],
            # check the name of different material and EoS at:
            # https://github.com/srbonilla/WoMa/blob/master/woma/misc/glob_vars.py
            A1_T_rho_type=["entropy=1750", "entropy=3027"],
            # Here I use fixed entropy for each layer, other option: "adiabatic","power=<float>"
            P_s=1e9,  # Surface pressure
            T_s=1500,  # Surface temperature
            M=impactor_mass * M_earth,
            R=R_earth * (impactor_mass ** 0.33)
        )

        impactor.gen_prof_L2_find_R1_given_M_R()

    except:

        print('Core/mantle ratio not viable, trying again')
        impactor.R = R_earth * (impactor_mass ** 0.2)
        impactor.gen_prof_L2_find_R1_given_M_R()

    if spin_period != 0:

        spin_target = woma.SpinPlanet(
            planet=target,
            name="target_spin_planet",
            period=spin_period  # h
        )

        print("Generating particles")
        target_particle_count = int(particle_count * (target_mass / total_mass))
        target_particles = woma.ParticlePlanet(spin_target, target_particle_count, verbosity=1)

    else:

        print("Generating particles")
        target_particle_count = int(particle_count * (target_mass / total_mass))
        target_particles = woma.ParticlePlanet(target, target_particle_count, verbosity=1)

    impactor_particle_count = int(particle_count * (impactor_mass / total_mass))
    impactor_particles = woma.ParticlePlanet(impactor, impactor_particle_count, verbosity=1)

    print("Calculating initial position and velocity of impactor\n")

    R_target = target.R
    R_impactor = impactor.R
    M_target = target.M
    M_impactor = impactor.M

    A1_pos_i, A1_vel_i = woma.impact_pos_vel_b_v_c_t(
        b=impact_parameter,
        v_c=impact_velocity,
        t=start_time,
        R_t=R_target,
        R_i=R_impactor,
        M_t=M_target,
        M_i=M_impactor,
        units_v_c=velocity_unit,
        units_b="b"
    )

    print(f'Target mass       : {M_target / M_earth}')
    print(f'Impactor mass     : {M_impactor / M_earth}')
    print(f'Impactor position : {A1_pos_i}')
    print(f'Impactor velocity : {A1_vel_i}\n')

    print('Writing initial conditions to impactor particles')

    # setting speed of impactor
    impactor_particles.A2_pos += A1_pos_i
    impactor_particles.A2_vel += A1_vel_i

    # moving to ZMF

    A1_vel_COM = (M_impactor * A1_vel_i) / (M_target + M_impactor)

    target_particles.A2_vel -= A1_vel_COM
    impactor_particles.A2_vel -= A1_vel_COM

    print('Combining particles into single file')

    import h5py
    with h5py.File(f'{working_directory}{name}/{name}_initial_conditions.hdf5', "w") as f:
        woma.save_particle_data(
            f,
            np.append(target_particles.A2_pos, impactor_particles.A2_pos, axis=0),
            np.append(target_particles.A2_vel, impactor_particles.A2_vel, axis=0),
            np.append(target_particles.A1_m, impactor_particles.A1_m),
            np.append(target_particles.A1_h, impactor_particles.A1_h),
            np.append(target_particles.A1_rho, impactor_particles.A1_rho),
            np.append(target_particles.A1_P, impactor_particles.A1_P),
            np.append(target_particles.A1_u, impactor_particles.A1_u),
            np.append(target_particles.A1_mat_id, impactor_particles.A1_mat_id),
            boxsize=100 * R_earth,
            file_to_SI=woma.Conversions(M_earth, R_earth, 1),
        )

    final_particle_count = len(np.append(target_particles.A1_m, impactor_particles.A1_m))

    data = [
        f'Target mass           :{M_target / M_earth:.4f} Mearth\n',
        f'Target spin period    :{spin_period} hours\n',
        f'Impactor mass         :{M_target / M_earth:.4f} Mearth\n',
        f'Mass ratio            :{mass_ratio:.4f}\n'
        f'Impact velocity       :{A1_vel_i} m/s ({impact_velocity:.1f} v_esc)\n',
        f'Impact parameter      :{impact_parameter}\n',
        f'Total particles       :{final_particle_count}\n'
    ]

    with open(f'{working_directory}{name}/{name}.txt', 'w') as file:
        file.writelines(data)

    print("Initial conditions saved")


def create_sim_directory(name):
    try:
        os.mkdir(f'{working_directory}{name}')
        print("Working directory created ")
    except FileExistsError:
        print("Working directory already exists")

    try:
        os.mkdir(f'{working_directory}{name}/output')
    except FileExistsError:
        pass

    shutil.copyfile('initial_condition_files/ANEOS_forsterite_S19.txt',
                    f'{working_directory}{name}/ANEOS_forsterite_S19.txt')
    shutil.copyfile('initial_condition_files/ANEOS_iron_S20.txt', f'{working_directory}{name}/ANEOS_iron_S20.txt')
    shutil.copyfile('initial_condition_files/resub.sh', f'{working_directory}{name}/resub.sh')
    os.system(f'chmod +x {working_directory}{name}/resub.sh')


def create_parameter_file(name, sim_time=72000, snapshot_frequency=600):
    with open('initial_condition_files/simulation_parameters.yml', 'r') as file:
        data = file.readlines()

    data[10] = f'    file_name:  {name}_initial_conditions.hdf5     # The initial conditions file to read\n'
    data[16] = f'    time_end:       {sim_time}                       # The end time of the simulation (in internal units).\n'
    data[24] = f'    delta_time:         {snapshot_frequency}                     # Time difference between consecutive outputs (in internal units)\n'

    with open(f'{working_directory}{name}/simulation_parameters.yml', 'w') as file:
        file.writelines(data)


def create_slurm_script(name):
    with open('initial_condition_files/slurm_template.sh', 'r') as file:
        data = file.readlines()

    file.close()

    data[2] = f'#SBATCH -J {name}\n'

    with open(f'{working_directory}{name}/impact_slurm.sh', 'w') as file:
        file.writelines(data)

    file.close()


def create_restart(name):
    with open('initial_condition_files/slurm_restart.sh', 'r') as file:
        data = file.readlines()

    file.close()

    data[2] = f'#SBATCH -J {name}_restart\n'

    with open(f'{working_directory}{name}/slurm_restart.sh', 'w') as file:
        file.writelines(data)

    file.close()
    

def create_preset_conditions():

    P = 1e5
    M = [0.5, 0.1, 0.25, 1, 2]
    K = [1, 0.5, 0.2, 0.05]
    V = [1.1, 3, 5]
    B = [0.5, 0.1, 0.3, 0.8]

    for m in M:
        generate_initial_conditions(P, m, K[0], V[0], B[0])

    for k in K:
        generate_initial_conditions(P, M[0], k, V[0], B[0])

    for v in V:
        generate_initial_conditions(P, M[0], K[0], v, B[0])

    for b in B:
        generate_initial_conditions(P, M[0], K[0], V[0], b)


preset = input('Create preset simulations? (Y/n): ')

if preset == 'Y':
    create_preset_conditions()
else:
    target_mass = float(input('Enter target mass (in Earth masses): '))
    target_spin = float(input('Enter target spin period (in hours) [if 0 is entered, no spin will be implemented]: '))
    mass_ratio = float(input('Enter impactor target mass ratio: '))
    impact_velocity = float(input('Enter impact velocity (in v_esc): '))
    impact_parameter = float(input('Enter impact parameter: '))
    particles = int(input('Enter particle count: '))
    time_before = int(input('Enter simulation start time (number of seconds before impact): '))
    simulation_time = int(input('Enter simulation running time (in seconds): '))
    snapshot_freq = int(input('Enter time between snapshots (in seconds): '))

    print('Generating conditions...')
    generate_initial_conditions(particles, target_mass, mass_ratio, impact_velocity, impact_parameter,
                                spin_period=target_spin, start_time=time_before)
