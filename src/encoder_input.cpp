// Copyright 2026 LHiou
// SPDX-License-Identifier: Apache-2.0

#include <cerrno>
#include <ffmpeg_encoder_decoder/encoder.hpp>
#include <ffmpeg_encoder_decoder/utils.hpp>
#include <limits>
#include <sstream>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/hwcontext.h>
}

namespace ffmpeg_encoder_decoder
{
namespace
{
auto MakeFrame()
{
  auto release = [](AVFrame * frame) { av_frame_free(&frame); };
  return std::unique_ptr<AVFrame, decltype(release)>(av_frame_alloc(), release);
}

struct GraphEnds
{
  AVFilterInOut * inputs{nullptr};
  AVFilterInOut * outputs{nullptr};
  ~GraphEnds()
  {
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
  }
};

int AttachDevice(AVFilterGraph * graph, AVBufferRef * device)
{
  if (!device) {
    return 0;
  }
  for (unsigned i = 0; i < graph->nb_filters; ++i) {
    auto * filter = graph->filters[i];
    if (!filter->hw_device_ctx) {
      filter->hw_device_ctx = av_buffer_ref(device);
      if (!filter->hw_device_ctx) {
        return AVERROR(ENOMEM);
      }
    }
  }
  return 0;
}

int ParseChain(
  AVFilterGraph * graph, const std::string & chain, AVBufferRef * device, GraphEnds * ends)
{
  // Newer hwupload filters need their device before filter initialization.
#if LIBAVFILTER_VERSION_MAJOR >= 10
  AVFilterGraphSegment * segment = nullptr;
  int ret = avfilter_graph_segment_parse(graph, chain.c_str(), 0, &segment);
  if (ret >= 0) {
    ret = avfilter_graph_segment_create_filters(segment, 0);
  }
  if (ret >= 0) {
    ret = AttachDevice(graph, device);
  }
  if (ret >= 0) {
    ret = avfilter_graph_segment_apply(segment, 0, &ends->inputs, &ends->outputs);
  }
  avfilter_graph_segment_free(&segment);
#else
  int ret = avfilter_graph_parse2(graph, chain.c_str(), &ends->inputs, &ends->outputs);
  if (ret >= 0) {
    ret = AttachDevice(graph, device);
  }
#endif
  return ret;
}
}  // namespace

bool Encoder::checkError(int error, const char * action) const
{
  if (error >= 0) {
    return true;
  }
  RCLCPP_ERROR_STREAM(logger_, action << ": " << utils::err(error));
  return false;
}

void Encoder::setInputFormat(const std::string & format)
{
  Lock lock(mutex_);
  inputFormat_ = format;
}

void Encoder::setHWDeviceName(const std::string & name)
{
  Lock lock(mutex_);
  deviceName_ = name;
}

void Encoder::setHWPixelFormat(const std::string & format)
{
  Lock lock(mutex_);
  hardwareFormat_ = format;
}

void Encoder::setConversionFilter(const std::string & chain)
{
  Lock lock(mutex_);
  conversionFilter_ = chain;
}

void Encoder::setEncoderColorSpace(const std::string & value)
{
  Lock lock(mutex_);
  colorSpace_ = value;
}

void Encoder::setEncoderColorRange(const std::string & value)
{
  Lock lock(mutex_);
  colorRange_ = value;
}

bool Encoder::openHwInput()
{
  const int width = codecContext_->width;
  const int height = codecContext_->height;
  if (width <= 0 || height <= 0 || width % 2 || height % 2) {
    RCLCPP_ERROR(logger_, "NV12 requires positive even dimensions");
    return false;
  }
  if (!deviceName_.empty()) {
    const auto type = av_hwdevice_find_type_by_name(deviceName_.c_str());
    if (type == AV_HWDEVICE_TYPE_NONE) {
      RCLCPP_ERROR_STREAM(logger_, "Unknown FFmpeg device: " << deviceName_);
      return false;
    }
    if (!checkError(
          av_hwdevice_ctx_create(&hwDeviceContext_, type, nullptr, nullptr, 0),
          "Open input device")) {
      return false;
    }
  }
  // A hardware filter supplies its own output pool when the graph is configured.
  if (hardwareFormat_.empty() || !conversionFilter_.empty()) {
    return true;
  }
  codecContext_->hw_frames_ctx = av_hwframe_ctx_alloc(hwDeviceContext_);
  if (!codecContext_->hw_frames_ctx) {
    return checkError(AVERROR(ENOMEM), "Allocate hardware frame pool");
  }
  auto * frames = reinterpret_cast<AVHWFramesContext *>(codecContext_->hw_frames_ctx->data);
  frames->format = codecContext_->pix_fmt;
  frames->sw_format = AV_PIX_FMT_NV12;
  frames->width = width;
  frames->height = height;
  frames->initial_pool_size = 20;  // Some devices require a fixed-size pool.
  return checkError(av_hwframe_ctx_init(codecContext_->hw_frames_ctx), "Initialize frame pool");
}

void Encoder::closeConversionFilter()
{
  avfilter_graph_free(&conversionGraph_);
  filterSource_ = nullptr;
  filterSink_ = nullptr;
}

bool Encoder::openConversionFilter()
{
  const bool software_default = conversionFilter_.empty();
  const std::string chain = software_default ? "scale=out_color_matrix=" + colorSpace_ +
                                                 ":out_range=" + colorRange_ + ",format=nv12"
                                             : conversionFilter_;
  conversionGraph_ = avfilter_graph_alloc();
  if (!conversionGraph_) {
    return checkError(AVERROR(ENOMEM), "Allocate conversion graph");
  }
  GraphEnds ends;
  if (!checkError(
        ParseChain(conversionGraph_, chain, hwDeviceContext_, &ends), "Parse conversion_filter")) {
    return false;
  }
  if (!ends.inputs || ends.inputs->next || !ends.outputs || ends.outputs->next) {
    RCLCPP_ERROR(logger_, "conversion_filter must have one input and one output");
    return false;
  }
  std::ostringstream args;
  args << "video_size=" << codecContext_->width << "x" << codecContext_->height
       << ":pix_fmt=bgr24:pixel_aspect=1/1:time_base=" << timeBase_.num << "/" << timeBase_.den;
  const auto * source = avfilter_get_by_name("buffer");
  const auto * sink = avfilter_get_by_name("buffersink");
  if (!source || !sink) {
    RCLCPP_ERROR(logger_, "FFmpeg lacks buffer or buffersink");
    return false;
  }
  if (
    !checkError(
      avfilter_graph_create_filter(
        &filterSource_, source, "ros_bgr", args.str().c_str(), nullptr, conversionGraph_),
      "Create BGR source") ||
    !checkError(
      avfilter_graph_create_filter(
        &filterSink_, sink, "encoded_input", nullptr, nullptr, conversionGraph_),
      "Create conversion sink") ||
    !checkError(
      avfilter_link(filterSource_, 0, ends.inputs->filter_ctx, ends.inputs->pad_idx),
      "Link conversion input") ||
    !checkError(
      avfilter_link(ends.outputs->filter_ctx, ends.outputs->pad_idx, filterSink_, 0),
      "Link conversion output") ||
    !checkError(avfilter_graph_config(conversionGraph_, nullptr), "Configure conversion graph")) {
    return false;
  }
  const auto * ref = av_buffersink_get_hw_frames_ctx(filterSink_);
  const auto * frames = ref ? reinterpret_cast<AVHWFramesContext *>(ref->data) : nullptr;
  const auto output_format = software_default ? AV_PIX_FMT_NV12 : codecContext_->pix_fmt;
  const bool nv12_output =
    output_format == AV_PIX_FMT_NV12 || (frames && frames->sw_format == AV_PIX_FMT_NV12);
  if (
    av_buffersink_get_format(filterSink_) != output_format ||
    av_buffersink_get_w(filterSink_) != codecContext_->width ||
    av_buffersink_get_h(filterSink_) != codecContext_->height || !nv12_output) {
    RCLCPP_ERROR(logger_, "Filter output must match the encoder dimensions and NV12 layout");
    return false;
  }
  if (ref) {
    // Fixed-pool encoders need the exact pool that produces their input frames.
    codecContext_->hw_frames_ctx = av_buffer_ref(ref);
    if (!codecContext_->hw_frames_ctx) {
      return checkError(AVERROR(ENOMEM), "Reference conversion frame pool");
    }
  }
  return true;
}

bool Encoder::drainPackets()
{
  int ret;
  do {
    ret = drainPacket(codecContext_->width, codecContext_->height);
  } while (ret == 0);
  return ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || checkError(ret, "Receive packet");
}

bool Encoder::uploadNv12Frame(AVFrame * frame)
{
  if (!codecContext_->hw_frames_ctx) {
    return true;
  }
  auto uploaded = MakeFrame();
  if (!uploaded) {
    return checkError(AVERROR(ENOMEM), "Allocate upload frame");
  }
  if (
    !checkError(
      av_hwframe_get_buffer(codecContext_->hw_frames_ctx, uploaded.get(), 0),
      "Allocate upload buffer") ||
    !checkError(av_hwframe_transfer_data(uploaded.get(), frame, 0), "Upload NV12")) {
    return false;
  }
  av_frame_unref(frame);
  av_frame_move_ref(frame, uploaded.get());
  return true;
}

bool Encoder::convertBgrFrame(AVFrame * frame)
{
  if (!filterSource_) {
    RCLCPP_ERROR(logger_, "BGR8 filter graph is not initialized");
    return false;
  }
  // buffersrc copies this non-refcounted ROS view before the caller can release it.
  frame->color_range = AVCOL_RANGE_JPEG;
  frame->colorspace = AVCOL_SPC_RGB;
  if (!checkError(av_buffersrc_write_frame(filterSource_, frame), "Send BGR frame to filter")) {
    return false;
  }
  av_frame_unref(frame);
  return checkError(av_buffersink_get_frame(filterSink_, frame), "Receive NV12 frame from filter");
}

void Encoder::encodeHwImage(
  const uint8_t * data, size_t size, size_t step, const std::string & format, const Header & header)
{
  rclcpp::Time start, converted;
  if (measurePerformance_) {
    start = rclcpp::Clock().now();
  }
  if (format != "nv12" && format != "bgr8") {
    RCLCPP_ERROR_STREAM(logger_, "Unsupported input encoding: " << format);
    return;
  }
  const bool nv12 = format == "nv12";
  const int width = codecContext_->width;
  const int height = codecContext_->height;
  const size_t row_bytes = static_cast<size_t>(width) * (nv12 ? 1 : 3);
  const size_t rows = static_cast<size_t>(height) + (nv12 ? height / 2 : 0);
  if (
    !data || step < row_bytes || step > std::numeric_limits<int>::max() || size < row_bytes ||
    (size - row_bytes) / step < rows - 1) {
    RCLCPP_ERROR(logger_, "Invalid input stride or truncated image");
    return;
  }
  auto frame = MakeFrame();
  if (!frame) {
    checkError(AVERROR(ENOMEM), "Allocate input frame");
    return;
  }
  frame->width = width;
  frame->height = height;
  frame->format = nv12 ? AV_PIX_FMT_NV12 : AV_PIX_FMT_BGR24;
  frame->data[0] = const_cast<uint8_t *>(data);
  frame->linesize[0] = static_cast<int>(step);
  const int64_t pts = pts_++;
  frame->pts = pts;
  if (nv12) {
    frame->data[1] = frame->data[0] + step * height;
    frame->linesize[1] = frame->linesize[0];
  } else if (!convertBgrFrame(frame.get())) {
    return;
  }
  if (frame->format == AV_PIX_FMT_NV12 && !uploadNv12Frame(frame.get())) {
    return;
  }
  frame->pts = pts;
  frame->color_range = codecContext_->color_range;
  frame->colorspace = codecContext_->colorspace;
  if (measurePerformance_) {
    converted = rclcpp::Clock().now();
    tdiffFrameCopy_.update((converted - start).seconds());
  }
  int ret = avcodec_send_frame(codecContext_, frame.get());
  if (ret == AVERROR(EAGAIN)) {
    if (!drainPackets()) {
      return;
    }
    ret = avcodec_send_frame(codecContext_, frame.get());
  }
  if (!checkError(ret, "Send input frame")) {
    return;
  }
  ptsToStamp_.emplace(pts, PTSMapEntry(header.stamp, header.frame_id));
  if (measurePerformance_) {
    tdiffSendFrame_.update((rclcpp::Clock().now() - converted).seconds());
    ++frameCnt_;
    totalInBytes_ += row_bytes * rows;
  }
  drainPackets();
  if (measurePerformance_) {
    tdiffTotal_.update((rclcpp::Clock().now() - start).seconds());
  }
}
}  // namespace ffmpeg_encoder_decoder
