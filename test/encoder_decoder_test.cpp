// -*-c++-*---------------------------------------------------------------------------------------
// Copyright 2025 Bernd Pfrommer <bernd.pfrommer@gmail.com>
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

#include <gtest/gtest.h>
#include <unistd.h>

#include "encoder_decoder_tester.hpp"

TEST(ffmpeg_encoder_decoder, encoder_decoder_bgr8_default_filter)
{
  EncoderDecoderTester tester;
  tester.setOriginalROSEncoding("bgr8");
  tester.setEncoder("libx264");
  tester.setEncoderColorRange("limited");
  tester.addAVOption("crf", "0");
  tester.setExpectedPacketEncoding("h264;nv12");
  tester.setFinalROSEncoding("bgr8");
  tester.setDecoder("h264");
  tester.setDecoderExpectedAVPixFmt("yuv420p");
  tester.setPixelTolerance(2);  // Allow rounding in the BGR8/NV12 round trip.
  tester.runTest();
  tester.check();
}

TEST(ffmpeg_encoder_decoder, encoder_decoder_mono)
{
  const std::string orig_enc = "mono8";
  EncoderDecoderTester tester;
  // encoding options
  tester.setOriginalROSEncoding(orig_enc);
  tester.setEncoder("libx265");
  tester.setCVBridgeTargetFormat(orig_enc);
  tester.setAVSourcePixelFormat("gray");
  tester.addAVOption("x265-params", "lossless=1");
  tester.setExpectedPacketEncoding("hevc;gray;" + orig_enc + ";" + orig_enc);
  tester.addAVOption("crf", "0");  // may not be needed for lossless
  // decoding options
  tester.setFinalROSEncoding(orig_enc);
  tester.setDecoder("hevc");
  tester.setDecoderExpectedAVPixFmt("gray");
  tester.runTest();
  tester.check();
}

TEST(ffmpeg_encoder_decoder, encoder_decoder_bayer)
{
  // tests the special hacks required for bayer encoding
  const std::string orig_enc = "bayer_rggb8";
  EncoderDecoderTester tester;
  // --- encoding options
  tester.setOriginalROSEncoding(orig_enc);
  tester.setEncoder("libx265");
  // setting target format no conversion
  tester.setCVBridgeTargetFormat(orig_enc);
  // encoder will automatically pick yuv420p,
  // then recognize that this requires single-channel to
  // color conversion, and do this losslessly.
  // tester.setAVSourcePixelFormat("yuv420p"); // not needed
  tester.addAVOption("x265-params", "lossless=1");
  tester.setExpectedPacketEncoding("hevc;yuv420p;" + orig_enc + ";" + orig_enc);
  // --- decoding options
  tester.setFinalROSEncoding(orig_enc);
  tester.setDecoder("hevc");
  tester.setDecoderExpectedAVPixFmt("yuv420p");
  tester.runTest();
  tester.check();
}

TEST(ffmpeg_encoder_decoder, auto_encodes_nv12_and_bgr8_without_a_configured_filter)
{
  ffmpeg_encoder_decoder::Encoder encoder;
  encoder.setEncoder("libx264");
  encoder.setGOPSize(1);
  encoder.setMaxBFrames(0);
  encoder.addAVOption("tune", "zerolatency");

  sensor_msgs::msg::Image image;
  image.width = 64;
  image.height = 48;
  image.encoding = "nv12";
  image.step = image.width;
  image.data.resize(image.step * image.height * 3 / 2, 128);
  unsigned packets = 0;
  ASSERT_TRUE(encoder.initialize(
    image.width, image.height,
    [&](
      const std::string &, const rclcpp::Time &, const std::string & encoding, uint32_t, uint32_t,
      uint64_t, uint8_t, uint8_t *, size_t) {
      EXPECT_EQ(encoding, "h264;nv12");
      ++packets;
    },
    image.encoding));

  encoder.encodeImage(image);
  image.encoding = "bgr8";
  image.step = image.width * 3;
  image.data.resize(image.step * image.height, 128);
  encoder.encodeImage(image);
  encoder.flush();
  EXPECT_EQ(packets, 2U);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
