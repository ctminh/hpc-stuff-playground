echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting MPICH on BEAST ..."
module use ~/.module
module use ~/loc-libs/spack/share/spack/lmod/linux-sles15-zen2
module load mpich-3.3.2-gcc-10.2.1-nfqpwl7  # without linking fortran

# load dependencies
echo "3. Loading thallium-rpc, mercury, margo dependencies..."
module load hwloc-2.4.1-gcc-10.2.1-bu232a3
module load cmake-3.20.3-gcc-10.2.1-crj7py7
module load argobots-1.1-gcc-10.2.1-s3e2vao
module load boost-1.76.0-gcc-10.2.1-hotyooi
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load libfabric-1.11.1-gcc-10.2.1-iy4glnm # this one is built with an updated name of rdma-core on beast
module load mercury-2.0.1rc3-gcc-10.2.1-qlbkvcr # this one is built with an updated name of rdma-core on beast
module load mochi-abt-io-0.5.1-gcc-10.2.1-fm7cjy2
module load mochi-margo-0.9.4-gcc-10.2.1-d2oqzan
module load mochi-thallium-0.7-gcc-10.2.1-sgig2wg
module load hcl-tl-tcp-1.0


# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
cmake -DHCL_ENABLE_THALLIUM_TCP=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
