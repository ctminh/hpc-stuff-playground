# load intel compiler on BEAST-ROME
echo "Exporting intel compilers..."
source ~/intel/oneapi/setvars.sh

# load taskflow headers
export INCLUDE=/home/ctminh/projects/taskflow:$INCLUDE
export CPATH=/home/ctminh/projects/taskflow:$CPATH
