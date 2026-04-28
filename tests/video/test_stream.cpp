// Copyright 2026 Konovalenko Stanislav and Hombosh Oleh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <optional>
#include <stdexcept>
#include <string>
#include <filesystem>

#include <gtest/gtest.h>
#include <Eigen/Eigen>

#include "video/stream.h" 

constexpr size_t VIDEO_TIME = 10;
constexpr size_t VIDEO_FPS = 30;

struct VidCfg {
    static constexpr int WIDTH = 1280;
    static constexpr int HEIGHT= 720;
    static constexpr int FPS = 30;
    static constexpr int DURATION = 10;
    std::string name;

    VidCfg() = delete;
    VidCfg(std::string vid_name) : name(std::move(vid_name)) {}
};

class StreamTest: public testing::Test {
private:
    std::string path_;
protected:
    StreamTest() : path_(generateVideo({"test"})), stream_(path_) {}
    // void TearDown() override { std::filesystem::remove(path_); }

    video::Stream stream_;  // NOLINT
private:
    static std::string generateVideo(const VidCfg &cfg) {
        std::string dircmd = "mkdir -p \"" + std::string(TEST_DATA_DIR) + "/\"";
        int res = std::system(dircmd.c_str());
        if (res != 0) {
            throw std::runtime_error("ffmpeg failed (exit " + std::to_string(res) + ")");
        }

        std::string out = std::string(TEST_DATA_DIR) + "/" + std::string(cfg.name) + ".mp4";
        std::string cmd =
            "ffmpeg -y -f lavfi -i testsrc=duration=" + std::to_string(VidCfg::DURATION) +
            ":size=" + std::to_string(VidCfg::WIDTH) + "x" + std::to_string(VidCfg::HEIGHT) +
            ":rate=" + std::to_string(VidCfg::FPS) +
            " -c:v libx264 -profile:v high -level 4.0 -pix_fmt yuv420p " + out;
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            throw std::runtime_error("ffmpeg failed (exit " + std::to_string(ret) + ")");
        }
        std::cout << "Creating video at: " << std::filesystem::absolute(out) << std::endl;
        return out;
    }
};

TEST_F(StreamTest, ThrowsOnBadConstruction) {
    std::string bad_path = "non_existent.mp4";
    EXPECT_THROW(video::Stream stream(bad_path), std::runtime_error)
        << "Should throw runtime error when input file does not exist";

    std::string not_a_video_path = "not_a_video.txt";
    EXPECT_THROW(video::Stream stream(not_a_video_path), std::runtime_error)
        << "Should throw runtime error when input file is not video";
}

TEST_F(StreamTest, TestFrameCount) {
    auto res = stream_.getFrame();
    size_t n_frames = 0;
    while (res != std::nullopt) {
        ++n_frames;
        res = stream_.getFrame();
    }

    EXPECT_EQ(n_frames, VIDEO_FPS * VIDEO_TIME)
        << "The number of MatrixXd should equal the number of frames in the video (check VIDEO_FPS and VIDEO_TIME)";
}

TEST_F(StreamTest, TestFrameDimensions) {
    auto frame = stream_.getFrame();
    ASSERT_TRUE(frame.has_value()) << "First frame should not be nullopt";

    EXPECT_EQ(frame->rows(), VidCfg::HEIGHT)
        << "Frame height should match VidCfg::HEIGHT";
    EXPECT_EQ(frame->cols(), VidCfg::WIDTH)
        << "Frame width should match VidCfg::WIDTH";
}

TEST_F(StreamTest, TestPixelValueRange) {
    auto frame = stream_.getFrame();
    ASSERT_TRUE(frame.has_value()) << "First frame should not be nullopt";

    double min_val = frame->minCoeff();
    double max_val = frame->maxCoeff();

    EXPECT_GE(min_val, 0.0) << "Minimum pixel value must be >= 0.0";
    EXPECT_LE(max_val, 1.0) << "Maximum pixel value must be <= 1.0";
}

TEST_F(StreamTest, TestFramesAreNotIdentical) {
    auto frame1 = stream_.getFrame();
    auto frame2 = stream_.getFrame();

    ASSERT_TRUE(frame1.has_value()) << "First frame should not be nullopt";
    ASSERT_TRUE(frame2.has_value()) << "Second frame should not be nullopt";

    // Compute L2 norm of the difference; should be nonzero for an animated source
    double diff = (*frame1 - *frame2).norm();
    EXPECT_GT(diff, 0.0) << "Consecutive frames from testsrc should differ";
}
