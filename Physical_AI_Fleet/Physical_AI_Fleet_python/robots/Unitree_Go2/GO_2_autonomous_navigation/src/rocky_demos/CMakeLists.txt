cmake_minimum_required(VERSION 3.10)
project(rocky_slam_nav)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(UNITREE_LIB_DIR /unitree/lib/unitree_slam/unitree_robotics/lib/aarch64)
set(UNITREE_INCLUDE_DIR /unitree/lib/unitree_slam/unitree_robotics/include)

include_directories(
    ${UNITREE_INCLUDE_DIR}
    ${UNITREE_INCLUDE_DIR}/ddscxx
)

file(GLOB ROCKY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src ${ROCKY_SOURCES})
    get_filename_component(exe_name ${src} NAME_WE)
    add_executable(${exe_name} ${src})
    target_link_libraries(${exe_name}
        ${UNITREE_LIB_DIR}/libunitree_sdk2.a
        ${UNITREE_LIB_DIR}/libunitree_ros2_idl_cpp.a
        ${UNITREE_LIB_DIR}/libddscxx.so
        ${UNITREE_LIB_DIR}/libddsc.so
        Threads::Threads
    )
endforeach()
