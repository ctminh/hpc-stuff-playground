echo "1. Loading dependencies (on BEAST)..."
# module load intel/19.1.1 # just to use mpirun
# module load openmpi/4.0.4-gcc10

module use ~/.module
module use ~/loc-libs/spack/share/spack/modules/linux-sles15-zen2

module load hwloc-2.4.1-gcc-10.2.1-7svevzl
# module load mpich-3.3.2-gcc-10.2.1-xoqrrzh  # linking fortran but got error on beast
module load mpich-3.3.2-gcc-10.2.1-fyao74s  # without linking fortran
module load cmake-3.20.1-gcc-10.2.1-7cjd5mz
module load argobots-1.1-gcc-10.2.1-s3e2vao
module load boost-1.76.0-gcc-10.2.1-h37ct6b
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load libfabric-1.11.1-gcc-10.2.1-7rkzvhv # this one is built with an updated name of rdma-core on beast
module load mercury-2.0.1rc3-gcc-10.2.1-565ptkn # this one is built with an updated name of rdma-core on beast
module load mochi-abt-io-0.5.1-gcc-10.2.1-rghdmos
module load mochi-margo-0.9.4-gcc-10.2.1-7sqzydv
module load mochi-thallium-0.7-gcc-10.2.1-hhkhxqk

## -----------------------------------------
## -------- Running server -----------------
export OMP_NUM_THREADS=1
echo "Run the server on ROME1..."
echo "   export OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "   mpirun -n 2 -ppn 1 --host rome1,rome2 ./tl_string_rdma_transf"
mpirun -n 2 -ppn 1 --host rome1,rome2 ./tl_string_rdma_transf