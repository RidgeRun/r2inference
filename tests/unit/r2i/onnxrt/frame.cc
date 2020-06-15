/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <r2i/r2i.h>
#include <r2i/onnxrt/frame.h>

#include <core/session/onnxruntime_cxx_api.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#define FRAME_WIDTH  2
#define FRAME_HEIGHT  2

TEST_GROUP (OnnxrtFrame) {
  std::shared_ptr<r2i::onnxrt::Frame> frame =
    std::make_shared<r2i::onnxrt::Frame>();
  std::shared_ptr<r2i::ImageFormat> format =
    std::make_shared<r2i::ImageFormat>(r2i::ImageFormat::Id::RGB);
  int width = FRAME_WIDTH;
  int height = FRAME_HEIGHT;
  float data[4] = {0.1, 0.2, 0.3, 0.4};
};

TEST (OnnxrtFrame, FrameConfigure) {
  r2i::RuntimeError error;

  error = frame->Configure(data, width, height, format->GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (OnnxrtFrame, FrameConfigureNullData) {
  r2i::RuntimeError error;

  error = frame->Configure(nullptr, width, height, format->GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (OnnxrtFrame, FrameConfigureNegativeWidth) {
  r2i::RuntimeError error;

  error = frame->Configure(data, -1, height, format->GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (OnnxrtFrame, FrameConfigureNegativeHeight) {
  r2i::RuntimeError error;

  error = frame->Configure(data, width, -1, format->GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (OnnxrtFrame, FrameGetData) {
  r2i::RuntimeError error;
  void *local_data;

  error = frame->Configure(data, width, height, format->GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_data = frame->GetData();
  POINTERS_EQUAL (data, local_data);
}

TEST (OnnxrtFrame, FrameGetWidth) {
  r2i::RuntimeError error;
  int local_width;

  error = frame->Configure(data, width, height, format->GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_width = frame->GetWidth();
  LONGS_EQUAL (width, local_width);
}

TEST (OnnxrtFrame, FrameGetHeight) {
  r2i::RuntimeError error;
  int local_height;

  error = frame->Configure(data, width, height, format->GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_height = frame->GetHeight();
  LONGS_EQUAL (height, local_height);
}

TEST (OnnxrtFrame, FrameGetFormat) {
  r2i::RuntimeError error;
  r2i::ImageFormat local_format;

  error = frame->Configure(data, width, height, format->GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_format = frame->GetFormat();
  LONGS_EQUAL (format->GetId(), local_format.GetId());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
