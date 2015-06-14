cmake_minimum_required(VERSION 2.8)
project( localgPbProject )

find_package(OpenCV)
IF(OpenCV_FOUND)
  add_executable( localgPb main.cxx )
  target_link_libraries(localgPb ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

