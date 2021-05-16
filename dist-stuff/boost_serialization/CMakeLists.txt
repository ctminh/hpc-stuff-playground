# require a minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(boost_serialization_demo)

# compile serialization_demo.cpp
add_executable(boost_serial_demo serialization_demo.cpp)
target_link_libraries(boost_serial_demo -lboost_serialization)

# compile boost_concepts.cpp
add_executable(boost_concepts boost_concepts.cpp)
target_link_libraries(boost_concepts -lboost_serialization)

# compile boost_serialize_obj_to_str.cpp
add_executable(boost_serialize_obj_to_str boost_serialize_obj_to_str.cpp)
target_link_libraries(boost_serialize_obj_to_str -lboost_serialization)