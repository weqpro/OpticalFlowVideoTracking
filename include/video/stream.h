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

#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <optional>
#include <string>

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;
}

namespace video {

struct FormatCtxDeleter {
    void operator()(AVFormatContext *ptr) const noexcept;
};
struct CodecCtxDeleter {
    void operator()(AVCodecContext *ptr) const noexcept;
};
struct FrameDeleter {
    void operator()(AVFrame *ptr) const noexcept;
};
struct PacketDeleter {
    void operator()(AVPacket *ptr) const noexcept;
};
struct SwsCtxDeleter {
    void operator()(SwsContext *ptr) const noexcept;
};

class Stream {
public:
    Stream() = delete;
    explicit Stream(const std::string &path);

    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
    Stream(Stream &&) = default;
    Stream &operator=(Stream &&) = default;
    ~Stream() = default;

    // Returns std::nullopt when the stream is exhausted.
    // Throws std::runtime_error on unrecoverable FFmpeg errors.
    std::optional<Eigen::MatrixXd> getFrame();

private:
    std::unique_ptr<AVFormatContext, FormatCtxDeleter> format_ctx_;
    std::unique_ptr<AVCodecContext, CodecCtxDeleter> codec_ctx_;
    std::unique_ptr<AVFrame, FrameDeleter> frame_;
    std::unique_ptr<AVPacket, PacketDeleter> packet_;
    std::unique_ptr<SwsContext, SwsCtxDeleter> sws_ctx_; // lazy-initialized

    int video_stream_index_{-1};
    bool flushing_{false};

    static void ffmpegCheck(int ret, const char *what);
    Eigen::MatrixXd frameToMatrix();
};

} // namespace video
