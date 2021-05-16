#!/bin/bash
#SBATCH -J bcl_f_queue
#SBATCH -o ./logs/fast_queue_test_%J.out
#SBATCH -e ./logs/fast_queue_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:05:00

module load slurm_setup

## Create a log folder for slurm output
LOGDIR="./logs"
if [ -d "$LOGDIR" ]; then  ## if exists already
    echo "${LOGDIR} is already exist..."
else    ## if not, create a new one
    mkdir ${LOGDIR}
fi

mpirun -n 2 ./fastqueue_pushpop
