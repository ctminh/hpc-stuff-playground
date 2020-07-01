# load hwloc on SuperMUC-NG
# module load hwloc

# CHAM_MODE
CHAM_HOME=/dss/dsshome1/lxc0D/ra56kop/task-based-models-survey/chameleon/install/cham_tool_itac

# export libffi & hwloc
# export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/lib/pkgconfig
# export LD_LIBRARY_PATH=/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/lib64:$LD_LIBRARY_PATH
# export CPATH=/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/include:/dss/dsshome1/lrz/sys/spack/release/19.1/opt/x86_avx512/hwloc/2.0.1-gcc-gir2kom/include:$CPATH

# export ch-libs
export LD_LIBRARY_PATH=$CHAM_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CHAM_HOME/lib:$LIBRARY_PATH
export INCLUDE=$CHAM_HOME/include:$INCLUDE
export CPATH=$CHAM_HOME/include:$CPATH
