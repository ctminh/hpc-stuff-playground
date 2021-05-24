export OMP_NUM_THREADS=3
mpirun -n 2 inspxe-cl -r inpsxe_result -collect ti1 -- ./taskStealing 500 10