cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME Optimize)
set(CMAKE_CXX_COMPILER icpc)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -m64 -pg -fopenmp -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fopenmp -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prof-gen=srcpos -prof-dir=/home/sylvex/Desktop/deepsjeng_631_bench/sylvex_ws/no_optimize/codecov -std=c++11 -fopenmp -O0")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prof-use -prof-dir=/home/sylvex/Desktop/deepsjeng_631_bench/sylvex_ws/no_optimize/codecov -std=c++11 -fopenmp -O0")
add_definitions(-DBIG_MEMORY)

project(${PROJECT_NAME})
# Add your source files
set(SOURCE_FILES
    src/sjeng.cpp
    src/initp.cpp
    src/draw.cpp
    src/bits.cpp
    src/board.cpp
    src/preproc.cpp
    src/see.cpp
    src/moves.cpp
    src/attacks.cpp
    src/state.cpp
    src/make.cpp
    src/epd.cpp
    src/endgame.cpp
    src/ttable.cpp
    src/pawn.cpp
    src/neval.cpp
    src/generate.cpp
    src/bitboard.cpp
    src/utils.cpp
    src/search.cpp
)

# Add your header files
set(HEADER_FILES
    include/attacks.h
    include/bitboard.h
    include/bits.h
    include/board.h
    include/config.h
    include/draw.h
    include/endgame.h
    include/epd.h
    include/evpar.h
    include/extvars.h
    include/generate.h
    include/initp.h
    include/make.h
    include/moves.h
    include/neval.h
    include/pawn.h
    include/preproc.h
    include/search.h
    include/see.h
    include/sjeng.h
    include/squares.h
    include/state.h
    include/ttable.h
    include/utils.h
)

add_executable(deepsjeng ${SOURCE_FILES} ${HEADER_FILES})

target_include_directories(deepsjeng PUBLIC include)