#!/bin/bash -l

#SBATCH -J test
#SBATCH -p icelake
#SBATCH --nodes 1
#SBATCH --cpus-per-task=70
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=pt426@cam.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --time=0-12:00:00

module load libtool-2.4.6-gcc-5.4.0-xtclmhg
module load fftw-3.3.6-pl2-gcc-5.4.0-rn4yojp
module load metis-5.1.0-gcc-5.4.0-63vpksi
module load gsl-2.3-gcc-5.4.0-lqjrzui
module load jemalloc-4.5.0-gcc-5.4.0-j3zbugm
module load python/3.8
module load hdf5/1.8.8

../swiftsim/swift -s -G -t 70 simulation_parameters.yml 2>&1 | tee output.log
