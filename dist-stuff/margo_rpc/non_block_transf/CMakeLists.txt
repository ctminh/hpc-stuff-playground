# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_nonblock_trans)

# compile the code
add_executable (server_nonblock_resp server_nonblock_resp.c)
target_link_libraries (server_nonblock_resp -lmercury -lmargo -labt)

add_executable (client_nonblock_call client_nonblock_call.c)
target_link_libraries (client_nonblock_call -lmercury -lmargo -labt)
