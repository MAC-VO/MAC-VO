cmake_minimum_required(VERSION 3.16)
project(macvo_orb_bow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

if(NOT ORB_SLAM3_ROOT)
  message(FATAL_ERROR "ORB_SLAM3_ROOT is required")
endif()

find_package(OpenCV 4 REQUIRED)
find_package(OpenSSL REQUIRED)

set(DBOW2_ROOT "${ORB_SLAM3_ROOT}/Thirdparty/DBoW2")
add_executable(macvo_orb_bow
  main.cpp
  ${DBOW2_ROOT}/DBoW2/BowVector.cpp
  ${DBOW2_ROOT}/DBoW2/FORB.cpp
  ${DBOW2_ROOT}/DBoW2/FeatureVector.cpp
  ${DBOW2_ROOT}/DBoW2/ScoringObject.cpp
  ${DBOW2_ROOT}/DUtils/Random.cpp
  ${DBOW2_ROOT}/DUtils/Timestamp.cpp
)

target_include_directories(macvo_orb_bow PRIVATE
  ${ORB_SLAM3_ROOT}
  ${DBOW2_ROOT}
  ${OpenCV_INCLUDE_DIRS}
)
target_compile_options(macvo_orb_bow PRIVATE -O3 -Wall -Wextra)
target_link_libraries(macvo_orb_bow PRIVATE ${OpenCV_LIBS} OpenSSL::Crypto)
