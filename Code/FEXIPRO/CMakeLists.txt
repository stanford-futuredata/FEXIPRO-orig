cmake_minimum_required(VERSION 2.8)
project(FEXIPRO)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -std=c++0x -march=native -ftree-vectorize -lgfortran -fopenmp")
add_definitions(-DTIME_IT)
if(RULE)
  message(STATUS "Adding ONLINE_DECISION_RULE")
  add_definitions(-DONLINE_DECISION_RULE)
endif()

if(TEST_ONLY)
  message(STATUS "Adding TEST_ONLY")
  add_definitions(-DTEST_ONLY)
endif()

set(alg
        alg/Naive.h
        alg/svd/SVDIncrPrune.h
        alg/svd/SVDIntUpperBoundIncrPrune.h
        alg/svd/SVDIncrPruneIndividualReorder.h
        alg/tree/BallTreeSearch.h
        alg/tree/FastMKS.h
        alg/int/IntUpperBound.h
        alg/int/IntUpperBound2.h
        alg/svd/SVDIntUpperBoundIncrPrune2.h
        alg/transformation/TransformIncrPrune.h
        alg/transformation/TransformSVDIncrPrune.h
        alg/transformation/TransformSVDIncrPrune2.h
        alg/simd/SIMDIntUpperBound.h)

set(util
        util/Base.h
        util/Conf.h
        util/Monitor.h
        util/FileUtil.h
        util/Calculator.h
        util/Logger.h
        util/SVDUtil.h
        util/TransformUtil.h
        util/SIMDUtil.h)

set(structs
        structs/Matrix.h
        structs/ExtendMatrix.h
        structs/VectorElement.h
        structs/BasicBallTreeNode.h
        structs/BallTreeNode.h
        structs/ExtendMatrixRow.h
        structs/IntMatrixRow.h
        structs/SVDMatrixRow.h
        structs/SIRMatrixRow.h
        structs/SVDIntMatrixRow.h
        structs/SIMDIntMatrixRow.h
        structs/SVDMatrixRowIndividualReorder.h
        structs/FastHeap.h
        structs/RedSVDMatrixRow.h
        structs/RedSVDMatrixRow2.h)

link_directories("/usr/lib/openblas-base")

set(ARMADILLO_INCLUDE_DIRS "~/packages/include/aramdillo")
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Boost COMPONENTS serialization program_options)

if(NOT ${Boost_FOUND})
    SET(BOOST_ROOT ~/local) # default
    SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
    find_package(Boost COMPONENTS serialization program_options REQUIRED)
endif(NOT ${Boost_FOUND})

if(Boost_FOUND)
    message(STATUS "Boost found " ${Boost_LIBRARY_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
    message(STATUS "Boost_LIBRARY_DIRS in  " ${Boost_LIBRARY_DIRS})
    message(STATUS "Boost_INCLUDE_DIRS in  " ${Boost_INCLUDE_DIRS})
endif()

set(MLPACK_INCLUDE_DIR "~/packages/include")
set(MLPACK_LIBRARY "~/packages/lib/libmlpack.so")

find_path(MLPACK_INCLUDE_DIR
        NAMES core.hpp prereqs.hpp
        PATHS ${MLPACK_INCLUDE_DIR})

#find_library(MLPACK_LIBRARY
#        NAMES mlpack
#        PATHS ${MLPACK_LIBRARY_DIR})

if(MLPACK_LIBRARY)
    message(STATUS "MLPACK found " ${MLPACK_LIBRARY_DIR})
    include_directories(${MLPACK_INCLUDE_DIR})
endif()

add_library(lib SHARED ${alg} ${util} ${structs})
set_target_properties(lib PROPERTIES LINKER_LANGUAGE CXX)

add_executable(runFEXIPRO runFEXIPRO.cpp)
target_link_libraries(runFEXIPRO lib)
target_link_libraries(runFEXIPRO armadillo)
target_link_libraries(runFEXIPRO ${Boost_LIBRARIES})
target_link_libraries(runFEXIPRO ${MLPACK_LIBRARY})
target_link_libraries(runFEXIPRO openblas)