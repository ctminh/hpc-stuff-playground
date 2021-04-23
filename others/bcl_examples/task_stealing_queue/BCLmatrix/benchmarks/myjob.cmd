#!/bin/bash
#SBATCH -J one16N_gasnet
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ge49nuk2/bcl/benchmarks/%x.%j.%N.out
#SBATCH -D /dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ge49nuk2/bcl/examples/simple
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --get-user-env
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=nikolan00@gmail.com
#SBATCH --export=NONE
#SBATCH --time=00:05:00
module load slurm_setup
module load intel/19.0.5
module load intel-mpi/2019.8.254
mpiexec -n $SLURM_NTASKS ./taskStealing 400 800 1
