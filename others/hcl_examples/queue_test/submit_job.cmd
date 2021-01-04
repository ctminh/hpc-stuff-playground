#!/bin/sh
#SBATCH -J hcl_q_test
#SBATCH -o ./logs/hcl_queue_%J.out
#SBATCH -e ./logs/hcl_queue_%J.err
#SBATCH -D ./
#SBATCH --time=00:05:00
#SBATCH --get-user-env
#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup

# Run the program
mpirun -n $SLURM_NTASKS ./build/queue_test
