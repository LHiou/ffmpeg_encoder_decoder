#ifndef FFMPEG_ENCODER_DECODER__RGA_CONVERTER_HPP_
#define FFMPEG_ENCODER_DECODER__RGA_CONVERTER_HPP_

#include <cstdint>
#include <vector>
extern "C" {
#include <libavutil/frame.h>
}

namespace ffmpeg_encoder_decoder
{
// Synchronous librga/im2d conversion; errors never fall back to CPU scaling.
class RgaConverter
{
public:
  explicit RgaConverter(int cores = 3) : cores_(cores) {}
  void convert(const AVFrame * source, AVFrame * destination);

private:
  int cores_;
  std::vector<uint8_t> packedNv12_;
};
}  // namespace ffmpeg_encoder_decoder
#endif
