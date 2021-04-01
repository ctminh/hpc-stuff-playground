# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_send_recv_rpc)

# compile the code
add_executable (server_sum server_sum.c)
target_link_libraries (server_sum -lmargo -labt)

add_executable (client_call client_call.c)
target_link_libraries (client_call -lmargo -labt)
