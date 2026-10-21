cmake_minimum_required(VERSION 3.20)
project(Flawn VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -mavx2 -mfma -ffast-math -funroll-loops)
endif()

add_executable(flawn
    src/main.cpp
    src/lexer.cpp
    src/parser.cpp
    src/analyzer.cpp
    src/vm.cpp
    src/codegen.cpp
    src/pybridge.cpp
    src/autotune.cpp
)
