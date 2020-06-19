#!/bin/sh
#SBATCH -J nested_struct_mpi_comm
#SBATCH -o ./result_%J.out
#SBATCH -e ./result_%J.err
#SBATCH -D ./
#SBATCH --time=00:05:00
#SBATCH --get-user-env
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup
export VT_LOGFILE_PREFIX=./itac_trace

# Run the program with cham_tool
mpirun -trace -n 2 /dss/dsshome1/0A/di49mew/chameleon_tool_dev/experiment/with-itac/examples/mpi_derived_data/nested_struct_mpi_comm
