#!/bin/bash
#SBATCH -J bcl_datarate
#SBATCH -o ./logs/bcl_datarate_test_%J.out
#SBATCH -e ./logs/bcl_datarate_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:30:00

module load slurm_setup

export OMP_NUM_THREADS=2
export VT_LOGFILE_PREFIX=/dss/dsshome1/lxc0D/ra56kop/hpc-stuff-playground/dist-stuff/bcl_examples/task_stealing_queue/BCLmatrix/examples/simple/traces

mpirun -n ${SLURM_NTASKS} -trace ./taskStealing 300 100 20

