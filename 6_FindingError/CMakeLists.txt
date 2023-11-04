cmake_minimum_required(VERSION 3.26)
project(SimpleNN C)

set(CMAKE_C_STANDARD 11)

add_executable(SimpleNN main.c
        simple_neural_networks.c
        simple_neural_networks.h)
