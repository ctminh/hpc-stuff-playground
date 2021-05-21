echo "1. Loading lib-dependencies, e.g., upcxx, ..."
echo "   Intel OneAPI is already loaded on CoolMuc2"
module use ~/.modules
# module load upcxx-2021.3.0

echo "2. Compiling the code..."
export BCL_BACKEND=MPI
export BACKEND=MPI
make
