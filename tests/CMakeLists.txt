add_executable(optical_flow_tests 
    test_stream.cpp
)

target_link_libraries(optical_flow_tests PRIVATE 
    core_lib 
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(optical_flow_tests)