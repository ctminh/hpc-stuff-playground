#!/bin/bash
#SBATCH -J bulkdat_rdma
#SBATCH -o ./logs/bulkdat_trans_test_%J.out
#SBATCH -e ./logs/bulkdat_trans_test_%J.err
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --export=NONE
#SBATCH --time=00:02:00

module load slurm_setup

module use ~/.modules
module load local-spack
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load argobots-1.0-gcc-7.5.0-x75podl
module load boost-1.75.0-gcc-7.5.0-xdru65d
module load cereal-1.3.0-gcc-7.5.0-jwb3bux
module load libfabric-1.11.1-gcc-7.5.0-p6j52ik
module load mercury-2.0.0-gcc-7.5.0-z55j3mp
module load mochi-abt-io-0.5.1-gcc-7.5.0-w7nm5r2
module load mochi-margo-0.9.1-gcc-7.5.0-n2p7v3n
module load mochi-thallium-0.8.4-gcc-7.5.0-u5zn3qg
## module load hcl-dev

## -----------------------------------------
## -------- Checking Allocated Nodes -------
echo "1. Checking assigned nodes for this job..."
scontrol show hostname ${SLURM_JOB_NODELIST} > nodelist.txt
nodelist_file="./nodelist.txt"
i=0
while IFS= read -r line
do  
    node_arr[$i]=$line
    let "i++"
done < "${nodelist_file}"

## -----------------------------------------
## -------- Running server -----------------
echo "2. Init the server on the 1st node..."
echo "   mpirun -n 1 --host ${node_arr[0]} ./rpc_server &"
mpirun -n 1 --host ${node_arr[0]} ./server_bulkdata_sum &

## -----------------------------------------
## -------- Running bash-script ------------
## read and split the chars
echo "-----[BASH-SCRIPT] Sleeping a while to let the server share addr..."
sleep 5

echo "-----[BASH-SCRIPT] Getting the server addr..."
cur_dir=$(pwd)
input_file=$(<${cur_dir}/f_server_addr.txt)
IFS=\/ read -a fields <<< $input_file
IFS=   read -a server_addr <<< $input_file

## reset IFS back the default
## set | grep ^IFS=

## fields now is an array with separate values
## echo "       + Print the array after reading with delimiter..."
## set | grep ^fields=\\\|^IN=
## e.g., fields=([0]="ofi+tcp" [1]="ofi_rxm://10.7.5.34:35271")
## echo "          fields[0] = ${fields[0]}"
## echo "          fields[1] = ${fields[1]}"
## echo "          fields[2] = ${fields[2]}"

## seperate IP_addr and Port_number
IFS=\: read -a sepa_addr <<< ${fields[2]}
echo "-----[BASH-SCRIPT] Server_IP_Addr=${sepa_addr[0]} | Port=${sepa_addr[1]}"

## -----------------------------------------
## -------- Running clients ----------------
echo "3. Running client (just work with the margo server_addr format)..."
echo "   mpirun -n 1 --host ${node_arr[1]} ./rpc_client ${server_addr}"
mpirun -n 1 --host ${node_arr[1]} ./client_call ${server_addr}

echo "Done!"
echo "--------------------------------------"
echo "--------------------------------------"

## -----------------------------------------
## -------- Remove tmp-stuff ---------------
rm ./nodelist.txt
rm ./f_server_addr.txt


