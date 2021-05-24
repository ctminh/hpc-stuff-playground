echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting Intel OneAPI on laptop mnm..."
module use /home/ctminh/projects/loc-libs/local-modulefiles
module use /home/ctminh/projects/loc-libs/spack/share/spack/modules/linux-ubuntu20.04-skylake
source /home/ctminh/intel/oneapi/setvars.sh

echo "3. Loading hcl, thallium dependencies, ..."
module load hwloc-2.3.0
module load argobots-1.0-gcc-9.3.0-o5iwjfp
module load boost-1.74.0-gcc-9.3.0-naro2r2
module load cereal-1.3.0-gcc-9.3.0-ebju5lb
module load libfabric-1.11.1-gcc-9.3.0-fi7vnoh
module load mercury-2.0.0-gcc-9.3.0-7e63np2
module load mochi-abt-io-0.4.1-gcc-9.3.0-5tenne2
module load mochi-margo-0.9-gcc-9.3.0-xvlpdeh
module load mochi-thallium-0.7-gcc-9.3.0-y4wyiep
module load hcl-tl-tcp-1.0

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
cmake -DHCL_ENABLE_THALLIUM_TCP=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} ..
