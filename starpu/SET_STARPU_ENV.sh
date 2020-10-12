# export dependencies for GPU-PC at MNM-Chair
OPENCL_INC=/usr/local/cuda/include
OPENCL_LIB=/usr/local/cuda/lib64
HWLOC_HOME=home/ctminh/Projects/loc-libs/hwloc/hwloc-2.2.0/build
export PKG_CONFIG_PATH=$HWLOC_HOME/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$OPENCL_LIB:$HWLOC_HOME/lib:$LD_LIBRARY_PATH
export INCLUDE=$OPENCL_INC:$HWLOC_HOME/include:$INCLUDE
export CPATH=$OPENCL_INC:$HWLOC_HOME/include:$CPATH

if [ "${EXPORT_MNM_GPU_PC}" = "1" ]
then
export STARTPU_HOME=/home/ctminh/Projects/starpu/starpu-1.3.5/install
export PKG_CONFIG_PATH=$STARTPU_HOME/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$STARTPU_HOME/lib:$LD_LIBRARY_PATH
export PATH=$PATH:$STARTPU_HOME/bin
export INCLUDE=$STARTPU_HOME/include/starpu/1.3:$INCLUDE
export CPATH=$STARTPU_HOME/include/starpu/1.3:$CPATH
fi
