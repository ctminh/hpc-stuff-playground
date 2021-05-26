#!/bin/bash
#SBATCH -J hcl_mxmtask_queue_test
#SBATCH -o ./logs/hcl_mxmtask_queue_test_gnu_%J.out
#SBATCH -e ./logs/hcl_mxmtask_queue_test_gnu_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --qos=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --export=NONE
#SBATCH --time=00:05:00

module load slurm_setup

module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load cmake-3.20.1-gcc-7.5.0-4dmivgm
module load argobots-1.0-gcc-7.5.0-mio7f7m
module load boost-1.76.0-gcc-7.5.0-fromrfo
module load cereal-1.3.0-gcc-7.5.0-jwb3bux
module load libfabric-1.11.1-gcc-7.5.0-yroqjwu
module load mercury-2.0.0-gcc-7.5.0-mmuxedi
module load mochi-abt-io-0.5.1-gcc-7.5.0-qo6dgid
module load mochi-margo-0.9.1-gcc-7.5.0-uw3savi
module load mochi-thallium-0.7-gcc-7.5.0-b6d73xq
module load hcl-tl-gnu-roce-1.0

## --------------------------------------------------------------------------
## -------- Running hcl with mpi --------------------------------------------
echo "mpirun -n ${SLURM_NTASKS} ./main"
mpirun -n ${SLURM_NTASKS} ./main


