// -*-c++-*---------------------------------------------------------------------------------------
// Copyright 2024 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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

#include <ffmpeg_encoder_decoder/encoder.hpp>

#ifdef USE_CV_BRIDGE_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <cerrno>
#include <cinttypes>
#include <ffmpeg_encoder_decoder/safe_param.hpp>
#include <ffmpeg_encoder_decoder/utils.hpp>
#include <fstream>
#include <iomanip>
#include <string>

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace ffmpeg_encoder_decoder
{
Encoder::Encoder() : logger_(rclcpp::get_logger("Encoder")) {}

Encoder::~Encoder()
{
  Lock lock(mutex_);
  closeCodec();
}

void Encoder::reset()
{
  Lock lock(mutex_);
  closeCodec();
}

void Encoder::setEncoder(const std::string & n)
{
  Lock lock(mutex_);
  encoder_ = n;
  codec_ = utils::find_codec(encoder_);
}

std::string Encoder::findCodec(const std::string & encoder) { return (utils::find_codec(encoder)); }

static void free_frame(AVFrame ** frame)
{
  if (*frame) {
    av_free(*frame);
    *frame = nullptr;
  }
}

void Encoder::closeCodec()
{
  closeConversionFilter();
  hwInput_ = false;
  usesHardwareFrames_ = false;
  ptsToStamp_.clear();
  if (codecContext_) {
    avcodec_free_context(&codecContext_);
    // avcodec_close(codecContext_);
    codecContext_ = nullptr;
  }
  free_frame(&frame_);
  free_frame(&hw_frame_);
  free_frame(&wrapperFrame_);

  if (packet_) {
    av_packet_free(&packet_);  // also unreferences the packet
    packet_ = nullptr;
  }
  if (hwDeviceContext_) {
    av_buffer_unref(&hwDeviceContext_);
  }
  if (swsContext_) {
    sws_freeContext(swsContext_);
    swsContext_ = NULL;
  }
}

AVPixelFormat Encoder::pixelFormat(const std::string & f) const
{
  if (f.empty()) {
    return (AV_PIX_FMT_NONE);
  }
  const auto fmt = av_get_pix_fmt(f.c_str());
  if (fmt == AV_PIX_FMT_NONE) {
    RCLCPP_ERROR_STREAM(logger_, "ignoring unknown pixel format: " << f);
  }
  return (fmt);
}

bool Encoder::initialize(int width, int height, Callback callback, const std::string & encoding)
{
  Lock lock(mutex_);
  closeCodec();
  callback_ = callback;
  return (openCodec(width, height, encoding));
}

void Encoder::openHardwareDevice(
  const AVCodec * codec, enum AVHWDeviceType hwDevType, int width, int height)
{
  int err = 0;
  err = av_hwdevice_ctx_create(&hwDeviceContext_, hwDevType, NULL, NULL, 0);
  utils::check_for_err("cannot create hw device context", err);

  AVBufferRef * hwframe_ctx_ref = av_hwframe_ctx_alloc(hwDeviceContext_);
  if (!hwframe_ctx_ref) {
    av_buffer_unref(&hwDeviceContext_);
    hwDeviceContext_ = 0;
    throw std::runtime_error("cannot allocate hwframe context!");
  }
  const auto hwframe_transfer_formats =
    utils::get_hwframe_transfer_formats(hwframe_ctx_ref, AV_HWFRAME_TRANSFER_DIRECTION_TO);

  if (hwframe_transfer_formats.empty()) {
    // Bernd: Apparently NVENC does not need a device to be opened at all.
    // But how to tell if a device should be opened in general?
    // Checking if the list of TO transfer formats is empty() works to tell
    // NVENC (no device required) from VAAPI (device required).
    // This test may be broken in general!
    av_buffer_unref(&hwDeviceContext_);
    hwDeviceContext_ = 0;
    av_buffer_unref(&hwframe_ctx_ref);
    RCLCPP_INFO_STREAM(logger_, "no need to open device for codec " << codec->name);
    return;
  }

  // cast the reference to a concrete pointer
  AVHWFramesContext * frames_ctx = reinterpret_cast<AVHWFramesContext *>(hwframe_ctx_ref->data);
  frames_ctx->format = utils::find_hw_config(&usesHardwareFrames_, hwDevType, codec);

  if (usesHardwareFrames_) {
    frames_ctx->sw_format =
      utils::get_preferred_pixel_format(usesHardwareFrames_, hwframe_transfer_formats);
    if (pixFormat_ != AV_PIX_FMT_NONE) {
      RCLCPP_INFO_STREAM(
        logger_, "user overriding software pix fmt " << utils::pix(frames_ctx->sw_format));
      RCLCPP_INFO_STREAM(logger_, "with " << utils::pix(pixFormat_));
      frames_ctx->sw_format = pixFormat_;  // override default at your own risk!
    }
    if (frames_ctx->sw_format == AV_PIX_FMT_NONE) {
      av_buffer_unref(&hwframe_ctx_ref);
      throw(std::runtime_error("cannot find valid sw pixel format!"));
    }
  }

  frames_ctx->width = width;
  frames_ctx->height = height;
  frames_ctx->initial_pool_size = 20;
  if ((err = av_hwframe_ctx_init(hwframe_ctx_ref)) < 0) {
    av_buffer_unref(&hwframe_ctx_ref);
    utils::throw_err("failed to initialize hardware frame context", err);
  }
  codecContext_->hw_frames_ctx = av_buffer_ref(hwframe_ctx_ref);
  av_buffer_unref(&hwframe_ctx_ref);

  if (codecContext_->hw_frames_ctx == nullptr) {
    throw(std::runtime_error("hardware decoder: cannot create buffer ref!"));
  }
}

bool Encoder::openCodec(int width, int height, const std::string & encoding)
{
  try {
    if (!doOpenCodec(width, height, encoding)) {
      closeCodec();
      return false;
    }
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR_STREAM(logger_, e.what());
    closeCodec();
    return (false);
  }
  RCLCPP_DEBUG_STREAM(
    logger_, "intialized codec " << encoder_ << " for image: " << width << "x" << height);
  return (true);
}

enum AVPixelFormat Encoder::findMatchingSourceFormat(
  const std::string & rosSrcFormat, enum AVPixelFormat targetFormat)
{
  const auto targetFormatDesc = av_pix_fmt_desc_get(targetFormat);
  if (!targetFormatDesc) {
    RCLCPP_ERROR_STREAM(logger_, "invalid libav target format: " << static_cast<int>(targetFormat));
    throw(std::runtime_error("invalid libav target format"));
  }
  // special hack to encode single-channel images as nv12/yuv420p etc
  if (utils::encode_single_channel_as_color(rosSrcFormat, targetFormat)) {
    return (targetFormat);
  }
  return (utils::ros_to_av_pix_format(rosSrcFormat));
}

bool Encoder::doOpenCodec(int width, int height, const std::string & origEncoding)
{
  int err = 0;
  if (inputFormat_ != "auto" && inputFormat_ != "bgr8" && inputFormat_ != "nv12") {
    RCLCPP_ERROR(logger_, "input_format must be auto, bgr8 or nv12");
    return false;
  }
  const auto & source_encoding = origEncoding.empty() ? cvBridgeTargetFormat_ : origEncoding;
  const auto & input_encoding = inputFormat_ == "auto" ? source_encoding : inputFormat_;
  hwInput_ = input_encoding == "bgr8" || input_encoding == "nv12";

  codecContext_ = nullptr;
  if (encoder_.empty()) {
    RCLCPP_ERROR(logger_, "No encoder set");
    return false;
  }
  if ((width % 32) != 0) {
    RCLCPP_WARN(logger_, "horiz res must be multiple of 32!");
  }
  if (encoder_ == "h264_nvmpi" && ((width % 64) != 0)) {
    RCLCPP_WARN(logger_, "horiz res must be multiple of 64!");
    return false;
  }
  // find codec
  const AVCodec * codec = avcodec_find_encoder_by_name(encoder_.c_str());
  if (!codec) {
    RCLCPP_ERROR_STREAM(logger_, "Cannot find encoder: " << encoder_);
    return false;
  }

  // allocate codec context
  codecContext_ = avcodec_alloc_context3(codec);
  if (!codecContext_) {
    return checkError(AVERROR(ENOMEM), "Allocate codec context");
  }

  auto pixFmts = utils::get_encoder_formats(codecContext_, codec);

  codecContext_->width = width;
  codecContext_->height = height;
  codecContext_->time_base = timeBase_;
  codecContext_->framerate = frameRate_;
  codecContext_->bit_rate = bitRate_;
  codecContext_->qmax = qmax_;  // 0: highest, 63: worst quality bound
  codecContext_->gop_size = GOPSize_;
  codecContext_->max_b_frames = maxBFrames_;  // nvenc can only handle zero!

  if (hwInput_) {
    if (pixFormat_ != AV_PIX_FMT_NONE && pixFormat_ != AV_PIX_FMT_NV12) {
      RCLCPP_ERROR(logger_, "BGR8/NV12 input requires pixel_format=nv12");
      return false;
    }
    if (colorSpace_ != "bt601" && colorSpace_ != "bt709") {
      RCLCPP_ERROR(logger_, "encoder_color_space must be bt601 or bt709");
      return false;
    }
    if (colorRange_ != "full" && colorRange_ != "limited") {
      RCLCPP_ERROR(logger_, "encoder_color_range must be full or limited");
      return false;
    }
    codecContext_->sw_pix_fmt = AV_PIX_FMT_NV12;
    codecContext_->pix_fmt =
      hardwareFormat_.empty() ? AV_PIX_FMT_NV12 : av_get_pix_fmt(hardwareFormat_.c_str());
    if (!hardwareFormat_.empty()) {
      const auto * desc = av_pix_fmt_desc_get(codecContext_->pix_fmt);
      if (!desc || !(desc->flags & AV_PIX_FMT_FLAG_HWACCEL) || deviceName_.empty()) {
        RCLCPP_ERROR(logger_, "hw_pixel_format requires a hardware format and hw_device");
        return false;
      }
    }
    codecContext_->color_range = colorRange_ == "limited" ? AVCOL_RANGE_MPEG : AVCOL_RANGE_JPEG;
    codecContext_->colorspace = colorSpace_ == "bt709" ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;
    if (!openHwInput()) {
      return false;
    }
    if ((inputFormat_ != "nv12" || !conversionFilter_.empty()) && !openConversionFilter()) {
      return false;
    }
  } else {
    const auto hwDevType = utils::find_hw_device_type(codec);
    if (hwDevType != AV_HWDEVICE_TYPE_NONE) {
      RCLCPP_INFO_STREAM(
        logger_, encoder_ << " uses hw device: " << av_hwdevice_get_type_name(hwDevType));
      openHardwareDevice(codec, hwDevType, width, height);
    }
    if (usesHardwareFrames_) {
      auto * frames_ctx = reinterpret_cast<AVHWFramesContext *>(codecContext_->hw_frames_ctx->data);
      codecContext_->sw_pix_fmt = frames_ctx->sw_format;
      codecContext_->pix_fmt = frames_ctx->format;
    } else {
      codecContext_->pix_fmt = (pixFormat_ != AV_PIX_FMT_NONE)
                                 ? pixFormat_
                                 : utils::get_preferred_pixel_format(usesHardwareFrames_, pixFmts);
      codecContext_->sw_pix_fmt = codecContext_->pix_fmt;
    }
  }
  avSourcePixelFormat_ = utils::pix(codecContext_->sw_pix_fmt);
  RCLCPP_INFO_STREAM(logger_, "using av_source_pixel_format: " << avSourcePixelFormat_);

  std::stringstream ss;
  for (const auto & kv : avOptions_) {
    setAVOption(kv.first, kv.second);
    ss << " " << kv.first << "=" << kv.second;
  }
  RCLCPP_INFO(
    logger_, "codec: %10s, bit_rate: %10" PRId64 " qmax: %2d options: %s", encoder_.c_str(),
    bitRate_, qmax_, ss.str().c_str());
  if (!hwInput_) {
    RCLCPP_INFO_STREAM(
      logger_, "cv_bridge_target_format: "
                 << cvBridgeTargetFormat_
                 << " libav: " << utils::pix(utils::ros_to_av_pix_format(cvBridgeTargetFormat_)));
  }
  RCLCPP_INFO_STREAM(logger_, "av_source_pixel_format: " << utils::pix(codecContext_->sw_pix_fmt));
  RCLCPP_INFO_STREAM(logger_, "encoder (hw) format:    " << utils::pix(codecContext_->pix_fmt));

  err = avcodec_open2(codecContext_, codec, NULL);
  if (!checkError(err, "Open codec")) {
    return false;
  }

  RCLCPP_INFO_STREAM(logger_, "opened codec: " << encoder_);
  packet_ = av_packet_alloc();
  if (!packet_) {
    return checkError(AVERROR(ENOMEM), "Allocate encoder packet");
  }
  if (hwInput_) {
    encoding_ = codec_ + ";nv12";
    return true;
  }
  frame_ = av_frame_alloc();
  if (!frame_) {
    throw(std::runtime_error("cannot alloc software frame!"));
  }
  if (usesHardwareFrames_) {
    hw_frame_ = av_frame_alloc();
    if (!hw_frame_) {
      throw(std::runtime_error("cannot alloc hardware frame!"));
    }
  }
  frame_->width = width;
  frame_->height = height;
  frame_->format = codecContext_->sw_pix_fmt;
  frame_->color_range = codecContext_->color_range;
  frame_->colorspace = codecContext_->colorspace;
  // allocate image for frame
  err = av_image_alloc(
    frame_->data, frame_->linesize, width, height, static_cast<AVPixelFormat>(frame_->format), 64);
  utils::check_for_err("cannot alloc image", err);

  if (usesHardwareFrames_) {
    err = av_hwframe_get_buffer(codecContext_->hw_frames_ctx, hw_frame_, 0);
    utils::check_for_err("cannot get hw frame buffer", err);
    if (!hw_frame_->hw_frames_ctx) {
      throw(std::runtime_error("no hardware frame context, out of mem?"));
    }
  }

  // create (src) frame that wraps the received raw image
  wrapperFrame_ = av_frame_alloc();
  wrapperFrame_->width = width;
  wrapperFrame_->height = height;
  wrapperFrame_->format = findMatchingSourceFormat(
    cvBridgeTargetFormat_, static_cast<enum AVPixelFormat>(frame_->format));

  // initialize format conversion library
  if (!swsContext_) {
    swsContext_ = sws_getContext(
      width, height, static_cast<AVPixelFormat>(wrapperFrame_->format),  // src
      width, height, static_cast<AVPixelFormat>(frame_->format),         // dest
      SWS_FAST_BILINEAR | SWS_ACCURATE_RND, NULL, NULL, NULL);
    if (!swsContext_) {
      throw(std::runtime_error("cannot allocate sws context"));
    }
  }
  encoding_ = codec_ + ";" + avSourcePixelFormat_ + ";" + cvBridgeTargetFormat_ + ";" +
              (origEncoding.empty() ? cvBridgeTargetFormat_ : origEncoding);
  return true;
}

void Encoder::setAVOption(const std::string & field, const std::string & value)
{
  if (!value.empty() && codecContext_ && codecContext_->priv_data) {
    const int err =
      av_opt_set(codecContext_->priv_data, field.c_str(), value.c_str(), AV_OPT_SEARCH_CHILDREN);
    if (err != 0) {
      RCLCPP_ERROR_STREAM(
        logger_, "cannot set option " << field << " to value " << value << ": " << utils::err(err));
    }
  }
}

void Encoder::encodeImage(const Image & msg)
{
  Lock lock(mutex_);
  if (hwInput_) {
    if (
      msg.width != static_cast<uint32_t>(codecContext_->width) ||
      msg.height != static_cast<uint32_t>(codecContext_->height)) {
      RCLCPP_ERROR(logger_, "Input resolution changed");
      return;
    }
    const auto & format = inputFormat_ == "auto" ? msg.encoding : inputFormat_;
    encodeHwImage(msg.data.data(), msg.data.size(), msg.step, format, msg.header);
    return;
  }
  if (msg.encoding == "nv12") {
    RCLCPP_ERROR(logger_, "Initialize the encoder for NV12 input");
    return;
  }
  rclcpp::Time t0;
  if (measurePerformance_) {
    t0 = rclcpp::Clock().now();
  }
  try {
    if (
      utils::encode_single_channel_as_color(
        cvBridgeTargetFormat_, static_cast<AVPixelFormat>(wrapperFrame_->format))) {
      // hack to encode single-channel ros formats as nv12/yuv420p etc
      cv::Mat img(
        msg.height, msg.width, cv::DataType<uint8_t>::type, const_cast<uint8_t *>(msg.data.data()),
        msg.step);
      // Add empty color channels to the bottom of matrix
      img.push_back(cv::Mat::zeros(msg.height / 2, msg.width, cv::DataType<uint8_t>::type));
      doEncodeImage(img, msg.header, t0);
    } else {
      cv::Mat img = cv_bridge::toCvCopy(msg, cvBridgeTargetFormat_)->image;
      doEncodeImage(img, msg.header, t0);
    }
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR_STREAM(logger_, "cv_bridge convert failed: " << e.what());
  }
  if (measurePerformance_) {
    const auto t1 = rclcpp::Clock().now();
    tdiffDebayer_.update((t1 - t0).seconds());
  }
}

void Encoder::encodeImage(const cv::Mat & img, const Header & header, const rclcpp::Time & t0)
{
  Lock lock(mutex_);
  if (hwInput_) {
    const bool bgr = img.type() == CV_8UC3;
    const auto format = bgr ? "bgr8" : "nv12";
    const int height = codecContext_->height;
    if (
      img.cols != codecContext_->width || (!bgr && img.type() != CV_8UC1) ||
      img.rows != (bgr ? height : height + height / 2) ||
      (inputFormat_ != "auto" && inputFormat_ != format)) {
      RCLCPP_ERROR(logger_, "Expected BGR8 or a stacked NV12 matrix matching the encoder");
      return;
    }
    encodeHwImage(img.data, img.dataend - img.data, img.step[0], format, header);
    return;
  }
  doEncodeImage(img, header, t0);
}

void Encoder::doEncodeImage(const cv::Mat & img, const Header & header, const rclcpp::Time & t0)
{
  rclcpp::Time t1, t2, t3;
  if (measurePerformance_) {
    frameCnt_++;
    t1 = rclcpp::Clock().now();
    totalInBytes_ += img.cols * img.rows;  // raw size!
  }
  // bend the memory pointers in colorFrame to the right locations
  av_image_fill_arrays(
    wrapperFrame_->data, wrapperFrame_->linesize, &(img.data[0]),
    static_cast<AVPixelFormat>(wrapperFrame_->format), wrapperFrame_->width, wrapperFrame_->height,
    1);
  sws_scale(
    swsContext_, wrapperFrame_->data, wrapperFrame_->linesize, 0, codecContext_->height,
    frame_->data, frame_->linesize);

  if (measurePerformance_) {
    t2 = rclcpp::Clock().now();
    tdiffFrameCopy_.update((t2 - t1).seconds());
  }

  frame_->pts = pts_++;
  ptsToStamp_.insert(PTSMap::value_type(frame_->pts, {header.stamp, header.frame_id}));

  int ret;
  if (usesHardwareFrames_) {
    ret = av_hwframe_transfer_data(hw_frame_, frame_, 0);  // from software -> hardware frame
    utils::check_for_err("error while copying frame to hw", ret);
    hw_frame_->pts = frame_->pts;
  }

  ret = avcodec_send_frame(codecContext_, usesHardwareFrames_ ? hw_frame_ : frame_);
  if (measurePerformance_) {
    t3 = rclcpp::Clock().now();
    tdiffSendFrame_.update((t3 - t2).seconds());
  }
  // now drain all packets
  while (ret == 0) {
    ret = drainPacket(img.cols, img.rows);
  }
  if (measurePerformance_) {
    const rclcpp::Time t4 = rclcpp::Clock().now();
    tdiffTotal_.update((t4 - t0).seconds());
  }
}

void Encoder::flush()
{
  Lock lock(mutex_);
  if (!codecContext_) {
    return;
  }
  int ret = avcodec_send_frame(codecContext_, nullptr);
  if (ret == AVERROR(EAGAIN) && drainPackets()) {
    ret = avcodec_send_frame(codecContext_, nullptr);
  }
  if (ret != AVERROR_EOF && checkError(ret, "Flush encoder")) {
    drainPackets();
  }
}

void Encoder::flush(const Header &) { flush(); }

int Encoder::drainPacket(int width, int height)
{
  rclcpp::Time t0, t1, t2;
  if (measurePerformance_) {
    t0 = rclcpp::Clock().now();
  }
  int ret = avcodec_receive_packet(codecContext_, packet_);
  if (measurePerformance_) {
    t1 = rclcpp::Clock().now();
    tdiffReceivePacket_.update((t1 - t0).seconds());
  }
  const AVPacket & pk = *packet_;
  if (ret == 0 && pk.size > 0) {
    if (measurePerformance_) {
      t2 = rclcpp::Clock().now();
      totalOutBytes_ += pk.size;
      tdiffCopyOut_.update((t2 - t1).seconds());
    }
    auto it = ptsToStamp_.find(pk.pts);
    if (it != ptsToStamp_.end()) {
      const auto & pe = it->second;
      callback_(pe.frame_id, pe.time, encoding_, width, height, pk.pts, pk.flags, pk.data, pk.size);
      if (measurePerformance_) {
        const auto t3 = rclcpp::Clock().now();
        tdiffPublish_.update((t3 - t2).seconds());
      }
      ptsToStamp_.erase(it);
    } else {
      RCLCPP_ERROR_STREAM(logger_, "pts " << pk.pts << " has no time stamp!");
    }
    av_packet_unref(packet_);  // free packet allocated by encoder
  }
  return (ret);
}

void Encoder::printTimers(const std::string & prefix) const
{
  Lock lock(mutex_);
  if (!frameCnt_) {
    return;
  }
  RCLCPP_INFO_STREAM(
    logger_, prefix << " pktsz: " << totalOutBytes_ / frameCnt_ << " compr: "
                    << totalInBytes_ / (double)totalOutBytes_ << " debay: " << tdiffDebayer_
                    << " fmcp: " << tdiffFrameCopy_ << " send: " << tdiffSendFrame_
                    << " recv: " << tdiffReceivePacket_ << " cout: " << tdiffCopyOut_
                    << " publ: " << tdiffPublish_ << " tot: " << tdiffTotal_);
}
void Encoder::resetTimers()
{
  Lock lock(mutex_);
  tdiffDebayer_.reset();
  tdiffFrameCopy_.reset();
  tdiffSendFrame_.reset();
  tdiffReceivePacket_.reset();
  tdiffCopyOut_.reset();
  tdiffPublish_.reset();
  tdiffTotal_.reset();
  frameCnt_ = 0;
  totalOutBytes_ = 0;
  totalInBytes_ = 0;
}
}  // namespace ffmpeg_encoder_decoder
