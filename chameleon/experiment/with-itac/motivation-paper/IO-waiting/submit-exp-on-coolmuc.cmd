#!/bin/bash
#SBATCH -J exp_io_waiting
#SBATCH -o ./results/exp_io_waiting_%J.out
#SBATCH -e ./results/exp_io_waiting_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --partition=mpp3_batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:05:00

module load slurm_setup
export OMP_NUM_THREADS=2

# Run the program with cham_tool
mpiexec -n 2 /dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/experiment/with-itac/motivation-paper/IO-waiting/exp_io_waiting_coolmuc 20 20
