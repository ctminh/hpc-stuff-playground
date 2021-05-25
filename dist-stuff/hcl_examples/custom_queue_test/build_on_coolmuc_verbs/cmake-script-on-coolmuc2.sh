echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Loading Intel OneAPI on CoolMUC2 ..."
# module load intel-oneapi/2021.1
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell

# load dependencies
echo "3. Loading thallium-rpc, mercury, margo dependencies..."
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load cmake-3.20.1-oneapi-2021.1-enbi76g
module load argobots-1.0-oneapi-2021.1-67jzgco
module load boost-1.76.0-oneapi-2021.1-z6jxqo3
module load cereal-1.3.0-oneapi-2021.1-h45mkqp
module load libfabric-1.11.1-oneapi-2021.1-ndxraku # this one is built with an updated name of rdma-core on coolmuc
module load mercury-2.0.0-oneapi-2021.1-gecxu7y # this one is built with an updated name of rdma-core on coolmuc
module load mochi-abt-io-0.5.1-oneapi-2021.1-byoahez
module load mochi-margo-0.9.1-oneapi-2021.1-fwjkhal
module load mochi-thallium-0.7-oneapi-2021.1-ln74gab

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
# cmake -DHCL_ENABLE_RPCLIB=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_MPI_CXX_COMPILER=${MPI_CXX} ..
cmake -DHCL_ENABLE_THALLIUM_ROCE=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_MPI_CXX_COMPILER=${MPI_CXX} ..
