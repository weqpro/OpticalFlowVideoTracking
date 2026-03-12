cmake_minimum_required(VERSION 3.16)
project(OpticalFlowVideoTracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Eigen ---
find_package(Eigen3 REQUIRED NO_MODULE)

# --- FFmpeg (libav) ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC  REQUIRED libavcodec)
pkg_check_modules(AVUTIL   REQUIRED libavutil)
pkg_check_modules(SWSCALE  REQUIRED libswscale)

# --- Executable ---
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/video/stream.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${AVFORMAT_INCLUDE_DIRS}
    ${AVCODEC_INCLUDE_DIRS}
    ${AVUTIL_INCLUDE_DIRS}
    ${SWSCALE_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${AVFORMAT_LIBRARY_DIRS}
    ${AVCODEC_LIBRARY_DIRS}
    ${AVUTIL_LIBRARY_DIRS}
    ${SWSCALE_LIBRARY_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWSCALE_LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    ${AVFORMAT_CFLAGS_OTHER}
    ${AVCODEC_CFLAGS_OTHER}
    ${AVUTIL_CFLAGS_OTHER}
    ${SWSCALE_CFLAGS_OTHER}
)
