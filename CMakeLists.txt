cmake_minimum_required(VERSION 3.16)
project(OpticalFlowVideoTracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Always generate compile_commands.json (needed by clang-tidy and clangd).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/ClangTidy.cmake)

# --- Eigen ---
find_package(Eigen3 REQUIRED NO_MODULE)

# --- FFmpeg (libav) ---
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
    
    set(FFMPEG_LIBS 
        PkgConfig::AVFORMAT 
        PkgConfig::AVCODEC 
        PkgConfig::AVUTIL 
        PkgConfig::SWSCALE
    )
endif()

# --- Executable ---
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/video/stream.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include 
    ${FFMPEG_INCLUDES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Eigen3::Eigen 
    ${FFMPEG_LIBS}
)

if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${FFMPEG_BASE_DIR}/bin" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
endif()

if(ENABLE_CLANG_TIDY)
  target_enable_clang_tidy(${PROJECT_NAME})
endif()