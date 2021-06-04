echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting MPICH on BEAST ..."
module use ~/.module
module use ~/loc-libs/spack/share/spack/modules/linux-sles15-zen2
module load mpich-3.3.2-gcc-10.2.1-fyao74s  # without linking fortran
# module load mpich-3.3.2-gcc-10.2.1-xoqrrzh  # linking fortran but got error on beast

echo "3. Loading hcl, thallium dependencies, ..."
module load hwloc-2.4.1-gcc-10.2.1-7svevzl
module load cmake-3.20.1-gcc-10.2.1-7cjd5mz
module load argobots-1.1-gcc-10.2.1-s3e2vao
module load boost-1.76.0-gcc-10.2.1-h37ct6b
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load libfabric-1.11.1-gcc-10.2.1-bba3lbj
module load mercury-2.0.1rc3-gcc-10.2.1-frvg6o6
module load mochi-abt-io-0.5.1-gcc-10.2.1-rghdmos
module load mochi-margo-0.9.4-gcc-10.2.1-caqzps7
module load mochi-thallium-0.7-gcc-10.2.1-i4yovp5 # with libfabric-1.11.1-gcc-10.2.1-bba3lbj enabled sockets
module load hcl-tl-roce-1.0

# module load libfabric-1.11.1-gcc-10.2.1-7rkzvhv # this one is built with an updated name of rdma-core on beast
# module load mercury-2.0.1rc3-gcc-10.2.1-565ptkn # this one is built with an updated name of rdma-core on beast
# module load mochi-margo-0.9.4-gcc-10.2.1-7sqzydv
# module load mochi-thallium-0.7-gcc-10.2.1-hhkhxqk

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
cmake -DHCL_ENABLE_THALLIUM_ROCE=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DENABLE_PARALLEL_OMP=1 \
    -DENABLE_GDB_DEBUG=1 \
    ..
