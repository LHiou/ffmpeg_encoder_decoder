#include <ffmpeg_encoder_decoder/rga_converter.hpp>
#include <stdexcept>
#include <string>
extern "C" {
#include <libavutil/imgutils.h>
}
#ifdef HAVE_RGA
#include <im2d.hpp>

namespace
{
rga_buffer_t wrap(const AVFrame * frame)
{
  int format, bytes = 1, heightStride = frame->height;
  switch (frame->format) {
    case AV_PIX_FMT_BGR24: format = RK_FORMAT_BGR_888; bytes = 3; break;
    case AV_PIX_FMT_RGB24: format = RK_FORMAT_RGB_888; bytes = 3; break;
    case AV_PIX_FMT_NV12: format = RK_FORMAT_YCbCr_420_SP; break;
    default: throw std::runtime_error("im2d supports BGR24/RGB24/NV12 in this converter");
  }
  if (!frame->data[0] || frame->width <= 0 || frame->height <= 0 ||
      frame->linesize[0] < frame->width * bytes || frame->linesize[0] % bytes) {
    throw std::runtime_error("Invalid im2d image stride or dimensions");
  }
  return wrapbuffer_virtualaddr_t(frame->data[0], frame->width, frame->height,
                                 frame->linesize[0] / bytes, heightStride, format);
}
}
#endif

namespace ffmpeg_encoder_decoder
{
void RgaConverter::convert(const AVFrame * source, AVFrame * destination)
{
#ifdef HAVE_RGA
  if (source->width != destination->width || source->height != destination->height) {
    throw std::runtime_error("im2d conversion requires matching image dimensions");
  }
  const bool toYuv = destination->format == AV_PIX_FMT_NV12;
  if ((source->format == AV_PIX_FMT_NV12) == toYuv) {
    throw std::runtime_error("im2d conversion requires RGB/BGR <-> NV12");
  }
  const auto nv12 = toYuv ? destination : source;
  if ((nv12->width % 2) || (nv12->height % 2)) {
    throw std::runtime_error("im2d NV12 dimensions must be even");
  }
  // FFmpeg may pad between planes. Pack only when im2d cannot use the original layout.
  const bool pack = nv12->linesize[0] != nv12->linesize[1] ||
    reinterpret_cast<uintptr_t>(nv12->data[1]) != reinterpret_cast<uintptr_t>(nv12->data[0]) +
      static_cast<size_t>(nv12->linesize[0]) * nv12->height;
  AVFrame packed{};
  if (pack) {
    packed.format = AV_PIX_FMT_NV12;
    packed.width = nv12->width;
    packed.height = nv12->height;
    const int size = av_image_get_buffer_size(AV_PIX_FMT_NV12, packed.width, packed.height, 16);
    if (size < 0) { throw std::runtime_error("Invalid NV12 buffer size"); }
    packedNv12_.resize(size);
    av_image_fill_arrays(packed.data, packed.linesize, packedNv12_.data(), AV_PIX_FMT_NV12,
                         packed.width, packed.height, 16);
    if (!toYuv && av_frame_copy(&packed, source) < 0) {
      throw std::runtime_error("Cannot pack NV12 source planes");
    }
  }
  auto src = wrap(pack && !toYuv ? &packed : source);
  auto dst = wrap(pack && toYuv ? &packed : destination);
  int mode;
  if (toYuv) {
    if (destination->colorspace == AVCOL_SPC_BT709) {
      if (destination->color_range == AVCOL_RANGE_JPEG) {
        throw std::runtime_error("im2d BT.709 full-range encoding is unsupported");
      }
      mode = IM_RGB_TO_YUV_BT709_LIMIT;
    } else {
      mode = destination->color_range == AVCOL_RANGE_MPEG ?
        IM_RGB_TO_YUV_BT601_LIMIT : IM_RGB_TO_YUV_BT601_FULL;
    }
  }
  if (!toYuv) {
    if (source->color_range == AVCOL_RANGE_JPEG) {
      if (source->colorspace == AVCOL_SPC_BT709) {
        throw std::runtime_error("im2d BT.709 full-range decoding is unsupported");
      }
      mode = IM_YUV_TO_RGB_BT601_FULL;
    } else {
      mode = source->colorspace == AVCOL_SPC_BT709 ?
        IM_YUV_TO_RGB_BT709_LIMIT : IM_YUV_TO_RGB_BT601_LIMIT;
    }
  }
  const auto configured = imconfig(IM_CONFIG_SCHEDULER_CORE, cores_);
  if (configured != IM_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("Cannot select RGA3 cores: ") + imStrError(configured));
  }
  const auto status = imcvtcolor(src, dst, src.format, dst.format, mode, 1);
  if (status != IM_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("im2d conversion failed: ") + imStrError(status));
  }
  if (pack && toYuv && av_frame_copy(destination, &packed) < 0) {
    throw std::runtime_error("Cannot copy NV12 output planes");
  }
  if (toYuv) {
    if (destination->colorspace != AVCOL_SPC_BT709) { destination->colorspace = AVCOL_SPC_BT470BG; }
    if (destination->color_range != AVCOL_RANGE_MPEG) { destination->color_range = AVCOL_RANGE_JPEG; }
  } else {
    destination->color_range = AVCOL_RANGE_JPEG;
    destination->colorspace = AVCOL_SPC_RGB;
  }
  destination->pts = source->pts;
#else
  (void)source;
  (void)destination;
  throw std::runtime_error("RGA conversion requested but built without librga/im2d");
#endif
}
}  // namespace ffmpeg_encoder_decoder
