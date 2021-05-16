#!/bin/bash

input="./nodelist.txt"
idx=0
while IFS= read -r line
do  
    node_arr[$idx]=$line
    let "idx++"
done < "${input}"

echo "node1: ${node_arr[0]}"
echo "node2: ${node_arr[1]}"
