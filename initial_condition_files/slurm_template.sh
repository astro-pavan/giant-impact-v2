#!/bin/bash -l

#SBATCH -J test
#SBATCH -p icelake
#SBATCH --nodes 1
#SBATCH --cpus-per-task=70
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=pt426@cam.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:00:00

module load hdf5/1.12.1
module load libtool-2.4.6-gcc-5.4.0-xtclmhg
module load fftw-3.3.6-pl2-gcc-5.4.0-rn4yojp
module load metis-5.1.0-gcc-5.4.0-pnrgiky
module load gsl-2.4-gcc-5.4.0-z4fspad
module load jemalloc-4.5.0-gcc-5.4.0-j3zbugm

../swiftsim/swift -s -G -t 70 simulation_parameters.yml 2>&1 | tee output.log
