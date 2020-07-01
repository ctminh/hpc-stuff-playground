#!/bin/bash
#SBATCH -J coolmuc_test_cham
#SBATCH -o ./results/ch-mxm_tool_%J.out
#SBATCH -e ./results/ch-mxm_tool_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
REM #SBATCH --clusters=cm2_tiny
REM #SBATCH --partition=cm2_tiny
#SBATCH --clusters=mpp3
#SBATCH --partition=mpp3_batch
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --export=NONE
#SBATCH --time=00:10:00
module load slurm_setup

##export VT_LOGFILE_PREFIX=/dss/dsshome1/0A/di49mew/chameleon_tool_dev/experiment/with-itac/cham_tool/

# Run the program with cham_tool
OMP_NUM_THREADS=1 CHAMELEON_TOOL=1 CHAMELEON_TOOL_LIBRARIES=1 mpirun -n 2 /dss/dsshome1/lxc0D/ra56kop/task-based-models-survey/chameleon/experiment/with-itac/cham_tool/mxm_unequal_tasks_tool_coolmuc 20 20
