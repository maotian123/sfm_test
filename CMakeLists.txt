#CMakeLists.txt
PROJECT(sfm_test)

find_package(Eigen3 REQUIRED)



set(ALL_TARGET_LIBRARIES "")
include(cmake/opencv.cmake)
include(cmake/ceres.cmake)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}

)

ADD_EXECUTABLE(sfm_node src/sfm_node.cpp)

target_link_libraries(sfm_node ${ALL_TARGET_LIBRARIES})
