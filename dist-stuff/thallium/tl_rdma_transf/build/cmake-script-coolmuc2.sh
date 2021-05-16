echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting Intel-MPI into LIBRARY_PATH, INCLUDE, etc, ..."
echo "   (on CoolMUC2, intel-oneapi/2021 is already loaded)..."

echo "3. Loading local-spack (on CoolMUC)..."
module use ~/.modules
module load local-spack
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell

# load dependencies
echo "4. Loading margo..."
module load cmake-3.19.5-gcc-7.5.0-bxubtlf
module load argobots-1.0-gcc-7.5.0-x75podl
module load boost-1.75.0-gcc-7.5.0-xdru65d
module load cereal-1.3.0-gcc-7.5.0-jwb3bux
module load libfabric-1.11.1-gcc-7.5.0-p6j52ik
module load mercury-2.0.0-gcc-7.5.0-z55j3mp
module load mochi-abt-io-0.5.1-gcc-7.5.0-w7nm5r2
module load mochi-margo-0.9.1-gcc-7.5.0-n2p7v3n
module load mochi-thallium-0.7-gcc-7.5.0-nbeiina


# indicate which compiler for C/C++
echo "5. Setting which C/C++ compiler is used..."
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc

# run cmake
echo "6. Running cmake to config..."
cmake -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
