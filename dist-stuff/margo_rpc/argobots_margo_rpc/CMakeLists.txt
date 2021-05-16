# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(margo_rpc_as_provider)

# compile the code

add_library (prov_alpha_client provider_alpha_client.c)
target_include_directories(prov_alpha_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library (prov_alpha_server provider_alpha_server.c)
target_include_directories(prov_alpha_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable (server server.c)
target_include_directories(server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (server prov_alpha_server -lmercury -lmargo -labt)

add_executable (client client.c)
target_include_directories(client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (client prov_alpha_client -lmercury -lmargo -labt)
