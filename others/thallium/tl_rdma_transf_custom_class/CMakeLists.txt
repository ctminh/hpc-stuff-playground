# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(thallium_rdma_transf_customclass)

# compile the code
add_executable(tl_rdma_customclass_server tl_rdma_customclass_server.cpp)
target_link_libraries(tl_rdma_customclass_server -lthallium -lmargo -lmercury -labt)

add_executable(tl_rdma_customclass_client tl_rdma_customclass_client.cpp)
target_link_libraries(tl_rdma_customclass_client -lthallium -lmargo -lmercury -labt)