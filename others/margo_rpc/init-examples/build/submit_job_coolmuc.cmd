#!/bin/bash
#SBATCH -J margo_init_test
#SBATCH -o ./logs/margo_init_test_%J.out
#SBATCH -e ./logs/margo_init_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:10:00

## load slurm-setups
module load slurm_setup

## load local-modules for rpc-services
module use ~/.modules
module load local-spack
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load argobots-1.0-gcc-7.5.0-x75podl
module load boost-1.75.0-gcc-7.5.0-xdru65d
module load cereal-1.3.0-gcc-7.5.0-jwb3bux
module load libfabric-1.11.1-gcc-7.5.0-p6j52ik
module load mercury-2.0.0-gcc-7.5.0-z55j3mp
module load mochi-abt-io-0.5.1-gcc-7.5.0-w7nm5r2
module load mochi-margo-0.9.1-gcc-7.5.0-n2p7v3n
module load mochi-thallium-0.8.4-gcc-7.5.0-u5zn3qg
module load hcl-dev

## run the programs
mpirun -n 1 ./margo_server : n -1 ./margo_client
