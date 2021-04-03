## list of address
ADDRESS="ofi+tcp;ofi_rxm://10.7.5.34:35271"

## sleep a while
echo "Sleeping a while before reading file..."
sleep 1

## read and split the chars
echo "Reading input_file..."
input_file=$(<f_server_addr.txt)
IFS=\/ read -a fields <<< $input_file ##"$ADDRESS"

## reset IFS back the default
## set | grep ^IFS=

## fields now is an array with separate values
echo "Print the array after reading with delimiter..."
set | grep ^fields=\\\|^IN=

## fields=([0]="ofi+tcp" [1]="ofi_rxm://10.7.5.34:35271")
echo "fields[0] = ${fields[0]}"
echo "fields[1] = ${fields[1]}"
echo "fields[2] = ${fields[2]}"

## get only IP-addr
IFS=\: read -a addr <<< ${fields[2]}
echo "IP_addr=${addr[0]} | Port=${addr[1]}"
