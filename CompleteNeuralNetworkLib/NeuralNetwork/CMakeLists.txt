cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralNetwork main.c
        utils.c
        utils.h
        nn.c
        nn.h
        nn.c)
