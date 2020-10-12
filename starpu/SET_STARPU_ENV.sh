# export for GPU-PC at MNM-Chair
if [ "${EXPORT_MNM_GPU_PC}" = "1" ]
then
export STARTPU_HOME=/home/ctminh/Projects/starpu/starpu-1.2.10/install
export PKG_CONFIG_PATH=$STARTPU_HOME/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$STARTPU_HOME/lib:$LD_LIBRARY_PATH
export PATH=$PATH:$STARTPU_HOME/bin
export INCLUDE=$STARTPU_HOME/include/starpu/1.2:$INCLUDE
export CPATH=$STARTPU_HOME/include/starpu/1.2:$CPATH

fi