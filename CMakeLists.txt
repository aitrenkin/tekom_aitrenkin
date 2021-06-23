cmake_minimum_required(VERSION 3.10)

project(tekom_aitrenkin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )

find_package(OpenMP QUIET)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS ${PROJECT_NAME}
)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "aitrenkin")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "9")
set(CPACK_PACKAGE_VERSION_PATCH "9")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "opencv")
set(CPACK_PACKAGE_DESCRIPTION "Simple app to merge original video frame, its RGB - histogram and overexposed frame")
set(CPACK_PACKAGE_CONTACT "Anton Trenkin aitrenkin@gmail.com")
INCLUDE(CPack)
