find_package(OpenCV REQUIRED)

add_executable(optical_flow_tests 
    test_stream.cpp
    ../src/optical_flow.cpp
)

target_include_directories(optical_flow_tests PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(optical_flow_tests PRIVATE 
    core_lib 
    GTest::gtest_main
    ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(optical_flow_tests)