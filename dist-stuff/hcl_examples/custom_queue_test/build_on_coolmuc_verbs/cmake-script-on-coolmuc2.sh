echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting Intel-MPI into LIBRARY_PATH, INCLUDE, etc, ..."
echo "   (on CoolMUC2, not sure neccessary on SuperMUC-NG)..."
export INCLUDE=/dss/dsshome1/lrz/sys/spack/staging/20.2.2/opt/x86_64/intel-mpi/2019.8.254-gcc-vyzek4m/compilers_and_libraries_2020.2.254/linux/mpi/intel64/include:$INCLUDE
export CPATH=/dss/dsshome1/lrz/sys/spack/staging/20.2.2/opt/x86_64/intel-mpi/2019.8.254-gcc-vyzek4m/compilers_and_libraries_2020.2.254/linux/mpi/intel64/include:$CPATH
export LIBRARY_PATH=/dss/dsshome1/lrz/sys/spack/staging/20.2.2/opt/x86_64/intel-mpi/2019.8.254-gcc-vyzek4m/compilers_and_libraries_2020.2.254/linux/mpi/intel64/lib:$LIBRARY_PATH

echo "3. Loading local-spack (on CoolMUC)..."
module use ~/.modules
module load local-spack
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell

# load dependencies
echo "3. Loading hcl, thallium, mercury-rpc dependencies..."
module load cmake-3.19.5-gcc-7.5.0-bxubtlf
module load argobots-1.0-gcc-7.5.0-x75podl
module load boost-1.75.0-gcc-7.5.0-xdru65d
module load cereal-1.3.0-gcc-7.5.0-jwb3bux
module load libfabric-1.11.1-gcc-7.5.0-p6j52ik
# module load mercury-2.0.0-gcc-7.5.0-z55j3mp
module load mercury-1.0.1-gcc-7.5.0-edeh2o7
module load mochi-abt-io-0.5.1-gcc-7.5.0-w7nm5r2
# module load mochi-margo-0.9.1-gcc-7.5.0-n2p7v3n
module load mochi-margo-0.5-gcc-7.5.0-wqceknu
# module load mochi-thallium-0.8.4-gcc-7.5.0-u5zn3qg
# module load mochi-thallium-0.7-gcc-7.5.0-nbeiina
module load mochi-thallium-0.5.4-gcc-7.5.0-6orqcsa
module load hcl-dev

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
# cmake -DHCL_ENABLE_RPCLIB=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_MPI_CXX_COMPILER=${MPI_CXX} ..
cmake -DHCL_ENABLE_THALLIUM_ROCE=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_MPI_CXX_COMPILER=${MPI_CXX} ..
