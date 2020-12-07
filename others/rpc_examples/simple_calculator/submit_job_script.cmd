#!/bin/sh
#SBATCH -J rpc_test
#SBATCH -o ./logs/rpc_test_%J.out
#SBATCH -e ./logs/rpc_test_%J.err
#SBATCH -D ./
#SBATCH --time=00:30:00
#SBATCH --get-user-env
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup

# Run the program with cham_tool
srun --multi-prog ser_clie_progs.conf