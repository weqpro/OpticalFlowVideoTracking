add_executable(optical_flow_tests 
    video/test_stream.cpp
)

target_link_libraries(optical_flow_tests PRIVATE
    core_lib
    GTest::gtest_main
)

target_compile_definitions(optical_flow_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_BINARY_DIR}/test_data"
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_data")

if(WIN32)
    target_include_directories(optical_flow_tests PRIVATE ${FFMPEG_INCLUDES})
endif()

include(GoogleTest)
gtest_discover_tests(optical_flow_tests)
