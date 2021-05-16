#!/bin/bash
#SBATCH -J bcl_datarate
#SBATCH -o ./logs/bcl_datarate_test_%J.out
#SBATCH -e ./logs/bcl_datarate_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --export=NONE
#SBATCH --time=00:30:00

module load slurm_setup

mpirun -n ${SLURM_NTASKS} ./dataRate 300

