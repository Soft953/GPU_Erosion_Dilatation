cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME gpgpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -std=c++17 -pedantic -g")

find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(SRC
    src/main.cc
)

link_directories(${CMAKE_SOURCE_DIR}/src)

add_executable (gpgpu ${SRC})

target_link_libraries( gpgpu ${OpenCV_LIBS} )