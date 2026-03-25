#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "video/stream.h" 

// TEST(НазваСутностіTest, ЩоРобимо_ЗаЯкихУмов_ЩоОчікуємо)
TEST(VideoStreamTest, Constructor_WhenFileDoesNotExist_ThrowsRuntimeError) {
    // 1. Arrange 
    std::string bad_path = "non_existent_fake_video.mp4";

    // 2 & 3. Act & Assert
    EXPECT_THROW({
        video::Stream stream(bad_path);
    }, std::runtime_error);
}