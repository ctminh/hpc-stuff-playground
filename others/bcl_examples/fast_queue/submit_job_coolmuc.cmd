#!/bin/bash
#SBATCH -J bcl_f_queue
#SBATCH -o ./logs/fast_queue_test_%J.out
#SBATCH -e ./logs/fast_queue_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:05:00

module load slurm_setup


mpirun -n 2 ./fastqueue_pushpop
