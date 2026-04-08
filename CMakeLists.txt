cmake_minimum_required(VERSION 3.16)
project(OpticalFlowVideoTracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/ClangTidy.cmake)

find_package(Eigen3 REQUIRED NO_MODULE)

if(WIN32)
    file(TO_CMAKE_PATH "$ENV{FFMPEG_PATH}" FFMPEG_BASE_DIR)
    find_path(AVFORMAT_INCLUDE_DIRS NAMES libavformat/avformat.h HINTS "${FFMPEG_BASE_DIR}/include")
    find_path(AVCODEC_INCLUDE_DIRS  NAMES libavcodec/avcodec.h   HINTS "${FFMPEG_BASE_DIR}/include")
    find_path(AVUTIL_INCLUDE_DIRS   NAMES libavutil/avutil.h    HINTS "${FFMPEG_BASE_DIR}/include")
    find_path(SWSCALE_INCLUDE_DIRS  NAMES libswscale/swscale.h   HINTS "${FFMPEG_BASE_DIR}/include")

    find_library(AVFORMAT_LIBRARIES NAMES avformat HINTS "${FFMPEG_BASE_DIR}/lib")
    find_library(AVCODEC_LIBRARIES  NAMES avcodec  HINTS "${FFMPEG_BASE_DIR}/lib")
    find_library(AVUTIL_LIBRARIES   NAMES avutil   HINTS "${FFMPEG_BASE_DIR}/lib")
    find_library(SWSCALE_LIBRARIES  NAMES swscale  HINTS "${FFMPEG_BASE_DIR}/lib")

    set(FFMPEG_LIBS ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVUTIL_LIBRARIES} ${SWSCALE_LIBRARIES})
    set(FFMPEG_INCLUDES ${AVFORMAT_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIRS} ${AVUTIL_INCLUDE_DIRS} ${SWSCALE_INCLUDE_DIRS})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
    pkg_check_modules(AVCODEC  REQUIRED IMPORTED_TARGET libavcodec)
    pkg_check_modules(AVUTIL   REQUIRED IMPORTED_TARGET libavutil)
    pkg_check_modules(SWSCALE  REQUIRED IMPORTED_TARGET libswscale)
    
    set(FFMPEG_LIBS PkgConfig::AVFORMAT PkgConfig::AVCODEC PkgConfig::AVUTIL PkgConfig::SWSCALE)
endif()

add_library(core_lib 
    src/video/stream.cpp
)

target_include_directories(core_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(core_lib PUBLIC 
    Eigen3::Eigen 
    ${FFMPEG_LIBS}
)

target_include_directories(core_lib PRIVATE ${FFMPEG_INCLUDES})

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE core_lib)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)

if(WIN32)
    add_custom_command(TARGET core_lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${FFMPEG_BASE_DIR}/bin"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    )
endif()

if(ENABLE_CLANG_TIDY)
  target_enable_clang_tidy(core_lib)
  target_enable_clang_tidy(${PROJECT_NAME})
endif()