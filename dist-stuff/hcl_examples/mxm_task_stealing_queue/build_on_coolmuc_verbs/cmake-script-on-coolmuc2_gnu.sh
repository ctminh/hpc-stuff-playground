echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt  ./CMakeFiles cmake_install.cmake  Makefile

echo "3. Loading local-spack (on CoolMUC)..."
module use ~/.modules
module load local-spack
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell

# load dependencies
echo "4. Loading margo..."
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

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
cmake -DHCL_ENABLE_THALLIUM_ROCE=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
