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

void FormatCtxDeleter::operator()(AVFormatContext* p) const noexcept {
    avformat_close_input(&p);
}

void CodecCtxDeleter::operator()(AVCodecContext* p) const noexcept {
    avcodec_free_context(&p);
}

void FrameDeleter::operator()(AVFrame* p) const noexcept {
    av_frame_free(&p);
}

void PacketDeleter::operator()(AVPacket* p) const noexcept {
    av_packet_free(&p);
}

void SwsCtxDeleter::operator()(SwsContext* p) const noexcept {
    sws_freeContext(p);
}

/// check for ffmpeg return error
/// throws std::runtime_error
void Stream::ffmpeg_check(int ret, const char* what) {
    if (ret < 0) {
        char buf[AV_ERROR_MAX_STRING_SIZE]{};
        av_strerror(ret, buf, sizeof(buf));
        throw std::runtime_error(std::string(what) + ": " + buf);
    }
}

/// constructor of videostream
Stream::Stream(const std::string& path) {
    AVFormatContext* raw_fmt = nullptr;
    ffmpeg_check(
        avformat_open_input(&raw_fmt, path.c_str(), nullptr, nullptr),
        "avformat_open_input"
    );
    format_ctx_.reset(raw_fmt);

    ffmpeg_check(
        avformat_find_stream_info(format_ctx_.get(), nullptr),
        "avformat_find_stream_info"
    );

    const AVCodec* decoder = nullptr;
    int idx = av_find_best_stream(
        format_ctx_.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0
    );
    ffmpeg_check(idx, "av_find_best_stream");
    if (!decoder)
        throw std::runtime_error("av_find_best_stream: no decoder found for stream");
    video_stream_index_ = idx;

    AVCodecContext* raw_codec = avcodec_alloc_context3(decoder);
    if (!raw_codec)
        throw std::runtime_error("avcodec_alloc_context3: out of memory");
    codec_ctx_.reset(raw_codec);

    ffmpeg_check(
        avcodec_parameters_to_context(
            codec_ctx_.get(),
            format_ctx_->streams[video_stream_index_]->codecpar
        ),
        "avcodec_parameters_to_context"
    );

    ffmpeg_check(
        avcodec_open2(codec_ctx_.get(), decoder, nullptr),
        "avcodec_open2"
    );

    AVFrame* raw_frame = av_frame_alloc();
    if (!raw_frame)
        throw std::runtime_error("av_frame_alloc: out of memory");
    frame_.reset(raw_frame);

    AVPacket* raw_packet = av_packet_alloc();
    if (!raw_packet)
        throw std::runtime_error("av_packet_alloc: out of memory");
    packet_.reset(raw_packet);
}

/// converts current decoded frame to grayscale MatrixXd
Eigen::MatrixXd Stream::frame_to_matrix() {
    const int width  = codec_ctx_->width;
    const int height = codec_ctx_->height;

    if (!sws_ctx_) { // lazy sws init
        SwsContext* raw_sws = sws_getContext(
            width, height, static_cast<AVPixelFormat>(frame_->format),
            width, height, AV_PIX_FMT_GRAY8,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        if (!raw_sws)
            throw std::runtime_error("sws_getContext: failed to create context");
        sws_ctx_.reset(raw_sws);
    }

    // Scale into a flat byte buffer.
    std::vector<uint8_t> gray_buf(static_cast<std::size_t>(width * height));
    uint8_t* dst_data[1]  = { gray_buf.data() };
    int      dst_stride[1] = { width };

    sws_scale(
        sws_ctx_.get(),
        frame_->data, frame_->linesize,
        0, height,
        dst_data, dst_stride
    );

    // Map the row-major byte buffer into Eigen, then cast to double in [0,1].
    using RowMajorMatrixXu8 = Eigen::Matrix<
        uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const RowMajorMatrixXu8> pixel_map(
        gray_buf.data(), height, width
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
                ffmpeg_check(
                    avcodec_send_packet(codec_ctx_.get(), nullptr),
                    "avcodec_send_packet (flush)"
                );
            } else {
                ffmpeg_check(read_ret, "av_read_frame");

                if (packet_->stream_index != video_stream_index_) {
                    av_packet_unref(packet_.get());
                    continue;
                }

                int send_ret = avcodec_send_packet(
                    codec_ctx_.get(), packet_.get()
                );
                av_packet_unref(packet_.get());
                ffmpeg_check(send_ret, "avcodec_send_packet");
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
            ffmpeg_check(recv_ret, "avcodec_receive_frame");

            return frame_to_matrix();
        }
    }
}

} // namespace video
