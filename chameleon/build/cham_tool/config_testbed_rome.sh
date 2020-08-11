# remove some previous compiled versions
rm -rf ./CMakeCache.txt  ./CMakeFiles  ./cmake_install.cmake ./Makefile  ./src

# Load itac
module load intel/19.1.1

# Export libffi
# SET PKG CONFIG with FFI
export PKG_CONFIG_PATH=/home/ra56kop/loc-libs/libffi-3.2.1/build/rome/lib/pkgconfig:/home/ra56kop/loc-libs/hwloc-2.2.0/install/rome/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/home/ra56kop/loc-libs/libffi-3.2.1/build/rome/lib64:/home/ra56kop/loc-libs/hwloc-2.2.0/install/rome/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/ra56kop/loc-libs/libffi-3.2.1/build/rome/lib64:/home/ra56kop/loc-libs/hwloc-2.2.0/install/rome/lib:$LIBRARY_PATH
export INCLUDE=/home/ra56kop/loc-libs/libffi-3.2.1/build/rome/lib/libffi-3.2.1/include:/home/ra56kop/loc-libs/hwloc-2.2.0/install/rome/include:$INCLUDE
export CPATH=/home/ra56kop/loc-libs/libffi-3.2.1/build/rome/lib/libffi-3.2.1/include:/home/ra56kop/loc-libs/hwloc-2.2.0/install/rome/include:$CPATH


export OMP_PLACES=cores
export OMP_PROC_BIND=close
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto
export I_MPI_FABRICS="shm:tmi"
export I_MPI_DEBUG=5
export KMP_AFFINITY=verbose

# Remember to set the link.txt in the folder chameleon.dir: add the path of libffi after -libff
# Add more 2 line in the file /CMakeModules/FFI.cmake in the chameleon source code
#set(FFI_INCLUDE_DIR "/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/include")
#set(FFI_LIBRARY_DIR "/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/lib64")

# Run cmake
cmake -DCMAKE_INSTALL_PREFIX=/home/ra56kop/task-based-models-survey/chameleon/install/rome   \
    -DCMAKE_C_COMPILER=/home/sw/stack/opt/x86_64/intel/19.1.1-gcc-zvv5bax/compilers_and_libraries_2020.1.217/linux/bin/intel64/icc    \
    -DCMAKE_CXX_COMPILER=/home/sw/stack/opt/x86_64/intel/19.1.1-gcc-zvv5bax/compilers_and_libraries_2020.1.217/linux/bin/intel64/icpc \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS='-DOFFLOAD_SEND_TASKS_SEPARATELY=1' \
    -DCMAKE_C_FLAGS='-DOFFLOAD_SEND_TASKS_SEPARATELY=1' \
    /home/ra56kop/task-based-models-survey/chameleon
