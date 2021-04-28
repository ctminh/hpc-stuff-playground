echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting Intel-MPI on BEAST system, etc, ..."
module load intel/19.1.1

# load dependencies
echo "3. Loading local-spack (on BEAST)..."
module use ~/.module
module use ~/loc-libs/spack/share/spack/modules/linux-sles15-zen2

# load dependencies
echo "4. Loading margo (on BEAST)..."
module load cmake-3.20.1-gcc-10.2.1-7cjd5mz
module load argobots-1.1-gcc-10.2.1-s3e2vao
module load boost-1.76.0-gcc-10.2.1-h37ct6b
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load libfabric-1.11.1-gcc-10.2.1-7rkzvhv # this one is built with an updated name of rdma-core on beast
module load mercury-2.0.1rc3-gcc-10.2.1-565ptkn # this one is built with an updated name of rdma-core on beast
module load mochi-abt-io-0.5.1-gcc-10.2.1-rghdmos
module load mochi-margo-0.9.4-gcc-10.2.1-7sqzydv
module load mochi-thallium-0.7-gcc-10.2.1-hhkhxqk


# indicate which compiler for C/C++
echo "5. Setting which C/C++ compiler is used..."
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc

# run cmake
echo "6. Running cmake to config..."
cmake -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
