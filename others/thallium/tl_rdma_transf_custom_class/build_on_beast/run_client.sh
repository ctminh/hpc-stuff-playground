echo "1. Loading dependencies (on BEAST)..."
module load intel/19.1.1 # just to use mpirun

module use ~/.module
module use ~/loc-libs/spack/share/spack/modules/linux-sles15-zen2

module load cmake-3.20.1-gcc-10.2.1-7cjd5mz
module load argobots-1.1-gcc-10.2.1-s3e2vao
module load boost-1.76.0-gcc-10.2.1-h37ct6b
module load cereal-1.3.0-gcc-10.2.1-vd6dtp3
module load libfabric-1.11.1-gcc-10.2.1-7rkzvhv # this one is built with an updated name of rdma-core on beast
module load mercury-2.0.1rc3-gcc-10.2.1-565ptkn # this one is built with an updated name of rdma-core on beast
module load mochi-abt-io-0.5.1-gcc-10.2.1-rghdmos
module load mochi-margo-0.9.4-gcc-10.2.1-7sqzydv
module load mochi-thallium-0.7-gcc-10.2.1-hhkhxqk

## -----------------------------------------
## -------- Running bash-script ------------
## read and split the chars

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
## -------- Running client -----------------
echo "2. Init the client on ROME2..."
echo "   mpirun -n 1 ./tl_rdma_customclass_client ${server_addr}"
mpirun -n 1 ./tl_rdma_customclass_client ${server_addr}


## -----------------------------------------
## -------- Remove tmp-stuff ---------------
rm ./f_server_addr.txt
