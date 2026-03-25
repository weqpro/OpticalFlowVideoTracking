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

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <stdexcept>
#include <string>
#include <vector>

#include "video/stream.h"

namespace video {

void FormatCtxDeleter::operator()(AVFormatContext* ptr) const noexcept {
    avformat_close_input(&ptr);
}

void CodecCtxDeleter::operator()(AVCodecContext* ptr) const noexcept {
    avcodec_free_context(&ptr);
}

void FrameDeleter::operator()(AVFrame* ptr) const noexcept {
    av_frame_free(&ptr);
}

void PacketDeleter::operator()(AVPacket* ptr) const noexcept {
    av_packet_free(&ptr);
}

void SwsCtxDeleter::operator()(SwsContext* ptr) const noexcept {
    sws_freeContext(ptr);
}

/// check for ffmpeg return error
/// throws std::runtime_error
void Stream::ffmpegCheck(int ret, const char* what) {
    if (ret < 0) {
        std::array<char, AV_ERROR_MAX_STRING_SIZE> buf{};
        av_strerror(ret, buf.data(), buf.size());
        throw std::runtime_error(std::string(what) + ": " + buf.data());
    }
}

/// constructor of videostream
Stream::Stream(const std::string& path) {
    AVFormatContext* raw_fmt = nullptr;
    ffmpegCheck(
        avformat_open_input(&raw_fmt, path.c_str(), nullptr, nullptr),
        "avformat_open_input"
    );
    format_ctx_.reset(raw_fmt);

    ffmpegCheck(
        avformat_find_stream_info(format_ctx_.get(), nullptr),
        "avformat_find_stream_info"
    );

    const AVCodec* decoder = nullptr;
    int idx = av_find_best_stream(
        format_ctx_.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0
    );
    ffmpegCheck(idx, "av_find_best_stream");
    if (decoder == nullptr) {
        throw std::runtime_error("av_find_best_stream: no decoder found for stream");
    }
    video_stream_index_ = idx;

    AVCodecContext* raw_codec = avcodec_alloc_context3(decoder);
    if (raw_codec == nullptr) {
        throw std::runtime_error("avcodec_alloc_context3: out of memory");
    }
    codec_ctx_.reset(raw_codec);

    ffmpegCheck(
        avcodec_parameters_to_context(
            codec_ctx_.get(),
            format_ctx_->streams[video_stream_index_]->codecpar // NOLINT(*-pro-bounds-pointer-arithmetic)
        ),
        "avcodec_parameters_to_context"
    );

    ffmpegCheck(
        avcodec_open2(codec_ctx_.get(), decoder, nullptr),
        "avcodec_open2"
    );

    AVFrame* raw_frame = av_frame_alloc();
    if (raw_frame == nullptr) {
        throw std::runtime_error("av_frame_alloc: out of memory");
    }
    frame_.reset(raw_frame);

    AVPacket* raw_packet = av_packet_alloc();
    if (raw_packet == nullptr) {
        throw std::runtime_error("av_packet_alloc: out of memory");
    }
    packet_.reset(raw_packet);
}

/// converts current decoded frame to grayscale MatrixXd
Eigen::MatrixXd Stream::frameToMatrix() {
    const int WIDTH  = codec_ctx_->width;
    const int HEIGHT = codec_ctx_->height;

    if (!sws_ctx_) { // lazy sws init
        SwsContext* raw_sws = sws_getContext(
            WIDTH, HEIGHT, static_cast<AVPixelFormat>(frame_->format),
            WIDTH, HEIGHT, AV_PIX_FMT_GRAY8,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        if (raw_sws == nullptr) {
            throw std::runtime_error("sws_getContext: failed to create context");
        }
        sws_ctx_.reset(raw_sws);
    }

    // Scale into a flat byte buffer.
    std::vector<uint8_t> gray_buf(static_cast<std::size_t>(WIDTH * HEIGHT));
    std::array<uint8_t*, 1> dst_data   = { gray_buf.data() };
    std::array<int, 1> dst_stride = { WIDTH };

    sws_scale(
        sws_ctx_.get(),
        frame_->data, frame_->linesize,
        0, HEIGHT,
        dst_data.data(), dst_stride.data()
    );

    // Map the row-major byte buffer into Eigen, then cast to double in [0,1].
    using RowMajorMatrixXu8 = Eigen::Matrix<
        uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const RowMajorMatrixXu8> pixel_map(
        gray_buf.data(), HEIGHT, WIDTH
    );
    return pixel_map.cast<double>() / 255.0;
}

// get the next frame, nullopt if stream exhausted
std::optional<Eigen::MatrixXd> Stream::getFrame() {
    while (true) {
        if (!flushing_) {
            int read_ret = av_read_frame(format_ctx_.get(), packet_.get());

            if (read_ret == AVERROR_EOF) {
                flushing_ = true;

                // null packet to signal end-of-stream to the decoder.
                ffmpegCheck(
                    avcodec_send_packet(codec_ctx_.get(), nullptr),
                    "avcodec_send_packet (flush)"
                );
            } else {
                ffmpegCheck(read_ret, "av_read_frame");

                if (packet_->stream_index != video_stream_index_) {
                    av_packet_unref(packet_.get());
                    continue;
                }

                int send_ret = avcodec_send_packet(
                    codec_ctx_.get(), packet_.get()
                );
                av_packet_unref(packet_.get());
                ffmpegCheck(send_ret, "avcodec_send_packet");
            }
        }

        // Drain all frames the decoder is ready to emit.
        while (true) {
            int recv_ret = avcodec_receive_frame(
                codec_ctx_.get(), frame_.get()
            );

            if (recv_ret == AVERROR(EAGAIN)) {
                // Decoder needs more input; break to outer loop.
                break;
            }
            if (recv_ret == AVERROR_EOF) {
                return std::nullopt; // stream exhausted
            }
            ffmpegCheck(recv_ret, "avcodec_receive_frame");

            return frameToMatrix();
        }
    }
}

} // namespace video
