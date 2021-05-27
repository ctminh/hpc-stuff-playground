#!/bin/bash
#SBATCH -J hcl_mxmtask_queue_test
#SBATCH -o ./logs/hcl_mxmtask_queue_test_oneapi_%J.out
#SBATCH -e ./logs/hcl_mxmtask_queue_test_oneapi_%J.err
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
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load cmake-3.20.1-oneapi-2021.1-enbi76g
module load argobots-1.0-oneapi-2021.1-67jzgco
module load boost-1.76.0-oneapi-2021.1-z6jxqo3
module load cereal-1.3.0-oneapi-2021.1-h45mkqp
module load libfabric-1.11.1-oneapi-2021.1-ndxraku
module load mercury-2.0.0-oneapi-2021.1-gecxu7y
module load mochi-abt-io-0.5.1-oneapi-2021.1-byoahez
module load mochi-margo-0.9.1-oneapi-2021.1-fwjkhal
module load mochi-thallium-0.7-oneapi-2021.1-ln74gab
module load hcl-tl-oneapi-roce-1.0

## --------------------------------------------------------------------------
## -------- Running hcl with mpi --------------------------------------------
echo "mpirun -n ${SLURM_NTASKS} ./main"
mpirun -n ${SLURM_NTASKS} ./main


