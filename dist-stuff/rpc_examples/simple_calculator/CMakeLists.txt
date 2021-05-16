# project name and cmake-requirement
cmake_minimum_required(VERSION 3.0.0)
project(calculator)

# find rpclib
find_library(RPC_LIB NAMES rpc PATHS /dss/dsshome1/0A/di49mew/loc-libs/rpclib-master/install/lib)
# find_package(rpclib REQUIRED PATHS /dss/dsshome1/0A/di49mew/loc-libs/rpclib-master/install)
# include_directories(${RPCLIB_INCLUDE_DIR})

# compile cal_server
add_executable(cal_server cal_server.cpp)
target_link_libraries(cal_server pthread "${RPC_LIB}")
set_target_properties(
    cal_server
    PROPERTIES
    CXX_STANDARD 14
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${RPCLIB_EXTRA_FLAGS}"
)
target_compile_definitions(cal_server PUBLIC ${RPCLIB_COMPILE_DEFINITIONS})

# compile cal_client
add_executable(cal_client cal_client.cpp)
target_link_libraries(cal_client pthread "${RPC_LIB}")
set_target_properties(
    cal_client
    PROPERTIES
    CXX_STANDARD 14
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${RPCLIB_EXTRA_FLAGS}"
)
target_compile_definitions(cal_client PUBLIC ${RPCLIB_COMPILE_DEFINITIONS})
