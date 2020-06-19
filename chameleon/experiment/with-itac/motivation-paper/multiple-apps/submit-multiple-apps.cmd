#!/bin/sh
#SBATCH -J multiple-apps
#SBATCH -o ./results/multiple-apps_%J.out
#SBATCH -e ./results/multiple-apps_%J.err
#SBATCH -D ./
#SBATCH --time=00:30:00
#SBATCH --get-user-env
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup

export IS_DISTRIBUTED=1
export IS_SEPARATE=0
export N_PROCS=2
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# export VT_LOGFILE_PREFIX=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/experiment/with-itac/motivation-paper/IO-waiting/results/itac_traces

# Run the program with cham_tool
OMP_NUM_THREADS=1 CHAMELEON_TOOL=1 CHAMELEON_TOOL_LIBRARIES=1 mpirun -n 2 /dss/dsshome1/0A/di49mew/chameleon_tool_dev/experiment/with-itac/motivation-paper/multiple-apps/multiple-apps 3000 3000
