# load hwloc on CoolMUC
module load hwloc/2.0

# CHAM_MODE
export CHAM_MODE=cham_tool

# export libffi & hwloc
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/dss/dsshome1/lxc0D/ra56kop/local_libs/libffi-3.3/build/lib/pkgconfig
export LD_LIBRARY_PATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libffi-3.3/build/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libffi-3.3/build/lib64:$LIBRARY_PATH
export INCLUDE=/dss/dsshome1/lxc0D/ra56kop/local_libs/libffi-3.3/build/include:$INCLUDE
export CPATH=/dss/dsshome1/lxc0D/ra56kop/local_libs/libffi-3.3/build/include:/dss/dsshome1/lrz/sys/spack/staging/20.1/opt/haswell/hwloc/2.0.2-gcc-bnl3wf4/include:$CPATH

# export ch-libs
export LD_LIBRARY_PATH=/dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/$CHAM_MODE/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/$CHAM_MODE/lib:$LIBRARY_PATH
export INCLUDE=/dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/$CHAM_MODE/include:$INCLUDE
export CPATH=/dss/dsshome1/lxc0D/ra56kop/chameleon_tool_dev/install/with_itac/$CHAM_MODE/include:$CPATH
