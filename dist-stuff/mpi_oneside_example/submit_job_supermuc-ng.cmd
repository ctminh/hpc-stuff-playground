#!/bin/sh
#SBATCH -J one_s_mpi
#SBATCH -o ./one_side_mpi_test_%J.out
#SBATCH -e ./one_side_mpi_test_%J.err
#SBATCH -D ./
#SBATCH --time=00:05:00
#SBATCH --get-user-env
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=24
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup

export VT_LOGFILE_PREFIX=/dss/dsshome1/0A/di49mew/task-based-models-survey/others/mpi_oneside_example/traces

# Run the program with cham_tool
mpirun -n 4 ./main
