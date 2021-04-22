#!/bin/bash
#SBATCH -J tl_rdma_transf_mpi_omp
#SBATCH -o ./logs/tl_rdmampiomp_test_%J.out
#SBATCH -e ./logs/tl_rdmampiomp_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=01:00:00

module load slurm_setup

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
module load mochi-thallium-0.7-gcc-7.5.0-nbeiina
## module load hcl-dev

## --------------------------------------------------------------------------
## -------- Running both tl-server and client in a single mpi-program -------
echo "mpirun -n ${SLURM_NTASKS} ./tl_mxm_rdma_trans"
mpirun -n ${SLURM_NTASKS} ./tl_mxm_rdma_transf


