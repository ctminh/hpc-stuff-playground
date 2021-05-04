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
module load hcl-dev

# Run the program
mpirun -n $SLURM_NTASKS ./build/queue_test
