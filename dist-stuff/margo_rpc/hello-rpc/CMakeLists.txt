# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_hello_rpc_example)

# compile the code
add_executable (rpc_server rpc-server.c)
target_link_libraries (rpc_server -lmargo -labt)

add_executable (rpc_client rpc-client.c)
target_link_libraries (rpc_client -lmargo -labt)
