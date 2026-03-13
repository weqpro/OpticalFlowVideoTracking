#pragma once

#include <memory>
#include <optional>
#include <string>
#include <Eigen/Eigen>

extern "C" {
    struct AVFormatContext;
    struct AVCodecContext;
    struct AVFrame;
    struct AVPacket;
    struct SwsContext;
}

namespace video {

struct FormatCtxDeleter { void operator()(AVFormatContext* ptr) const noexcept; };
struct CodecCtxDeleter  { void operator()(AVCodecContext* ptr)  const noexcept; };
struct FrameDeleter     { void operator()(AVFrame* ptr)         const noexcept; };
struct PacketDeleter    { void operator()(AVPacket* ptr)        const noexcept; };
struct SwsCtxDeleter    { void operator()(SwsContext* ptr)      const noexcept; };

class Stream {
public:
    Stream() = delete;
    explicit Stream(const std::string& path);

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) = default;
    Stream& operator=(Stream&&) = default;
    ~Stream() = default;

    // Returns std::nullopt when the stream is exhausted.
    // Throws std::runtime_error on unrecoverable FFmpeg errors.
    std::optional<Eigen::MatrixXd> getFrame();

private:
    std::unique_ptr<AVFormatContext, FormatCtxDeleter> format_ctx_;
    std::unique_ptr<AVCodecContext,  CodecCtxDeleter>  codec_ctx_;
    std::unique_ptr<AVFrame,         FrameDeleter>     frame_;
    std::unique_ptr<AVPacket,        PacketDeleter>    packet_;
    std::unique_ptr<SwsContext,      SwsCtxDeleter>    sws_ctx_;  // lazy-initialized

    int  video_stream_index_{-1};
    bool flushing_{false};

    static void ffmpegCheck(int ret, const char* what);
    Eigen::MatrixXd frameToMatrix();
};

} // namespace video
