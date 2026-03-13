cmake_minimum_required(VERSION 3.22...4.2)
project(lab3_ex2)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

add_executable(lab3_ex2 lab3_ex2.cpp)

target_link_libraries(
    lab3_ex2
    PRIVATE
        tt_metal
        ttnn
        Python3::Python
)

target_compile_definitions(lab3_ex2 PRIVATE OVERRIDE_KERNEL_PREFIX="${CMAKE_SOURCE_DIR}/")
