# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_init_example)

# compile the code
add_executable (margo_server margo_server.c)
target_link_libraries (margo_server -lmargo)

add_executable (margo_client margo_client.c)
target_link_libraries (margo_client -lmargo)