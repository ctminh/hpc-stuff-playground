# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_rdma_transf)

# compile the code
add_executable(tl_rdma_server tl_rdma_server.cpp)
target_link_libraries(tl_rdma_server -lthallium -lmargo -lmercury -labt)

add_executable(tl_rdma_client tl_rdma_client.cpp)
target_link_libraries(tl_rdma_client -lthallium -lmargo -lmercury -labt)