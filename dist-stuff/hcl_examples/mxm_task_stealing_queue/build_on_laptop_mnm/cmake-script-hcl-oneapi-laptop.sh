echo "1. Removing old-cmake-files..."
rm -r CMakeCache.txt ./CMakeFiles cmake_install.cmake  Makefile

echo "2. Exporting Intel OneAPI on laptop mnm..."
module use /home/ctminh/projects/loc-libs/local-modulefiles
module use /home/ctminh/projects/loc-libs/spack/share/spack/modules/linux-ubuntu20.04-skylake
source /home/ctminh/intel/oneapi/setvars.sh

echo "3. Loading hcl, thallium dependencies, ..."
module load hwloc-2.3.0
module load argobots-1.0-oneapi-2021.1-kqlazgb
module load boost-1.74.0-oneapi-2021.1-oyky634
module load cereal-1.3.0-oneapi-2021.1-yip4izv
module load libfabric-1.11.1-oneapi-2021.1-dcieza6
module load mercury-2.0.0-oneapi-2021.1-7ejilqd
module load mochi-abt-io-0.4.1-oneapi-2021.1-q7x35qe
module load mochi-margo-0.9-oneapi-2021.1-d4s7u3m
module load mochi-thallium-0.7-oneapi-2021.1-qwqzl4q
module load hcl-tl-tcp-oneapi-1.0

# indicate which compiler for C/C++
echo "4. Setting which C/C++ compiler is used..."
export C_COMPILER=mpicc
export CXX_COMPILER=mpicxx

# run cmake
echo "5. Running cmake to config..."
cmake -DHCL_ENABLE_THALLIUM_TCP=true -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    ..
