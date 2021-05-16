# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt  CMakeFiles  cmake_install.cmake  Makefile

# path to intel compiler
INTEL_COMPILER_HOME=/home/ctminh/intel/oneapi
C_COMP=icc #gcc
CXX_COMP=icpc #g++
FORT_COMP=ifort #gfortran

# load intel_compiler & itac
echo "Loading intel-compiler, boost-lib, ..."
source /home/ctminh/intel/oneapi/setvars.sh
module use ~/projects/loc-libs/local-modulefiles
module use ~/projects/loc-libs/spack/share/spack/modules/linux-ubuntu20.04-skylake
module load boost-1.74.0-gcc-9.3.0-naro2r2

# run cmake
cmake -DCMAKE_C_COMPILER=${C_COMP} -DCMAKE_CXX_COMPILER=${CXX_COMP} ..
