/* Copyright (C) 2019 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <r2i/tflite/frame.h>
#include <fstream>
#include <cstring>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#define FRAME_WIDTH  2
#define FRAME_HEIGHT  2

bool multiple_batches = false;
bool invalid_width = false;
bool invalid_height = false;
bool invalid_channels = false;
bool new_tensor_error = false;

TEST_GROUP (TfLiteFrame) {
  r2i::tflite::Frame frame;
  int width = FRAME_WIDTH;
  int height = FRAME_HEIGHT;
  float data[4] = {0.1, 0.2, 0.3, 0.4};
  r2i::ImageFormat format;

  void setup () {
    frame = r2i::tflite::Frame();
    format = r2i::ImageFormat(r2i::ImageFormat::Id::RGB);
  }

  void teardown () {
  }
};

TEST (TfLiteFrame, FrameConfigure) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TfLiteFrame, FrameConfigureNullData) {
  r2i::RuntimeError error;

  error = frame.Configure(nullptr, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TfLiteFrame, FrameConfigureNegativeWidth) {
  r2i::RuntimeError error;

  error = frame.Configure(data, -1, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TfLiteFrame, FrameConfigureNegativeHeight) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, -1, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TfLiteFrame, FrameGetData) {
  r2i::RuntimeError error;
  void *local_data;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_data = frame.GetData();
  POINTERS_EQUAL (data, local_data);
}

TEST (TfLiteFrame, FrameGetWidth) {
  r2i::RuntimeError error;
  int local_width;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_width = frame.GetWidth();
  LONGS_EQUAL (width, local_width);
}

TEST (TfLiteFrame, FrameGetHeight) {
  r2i::RuntimeError error;
  int local_height;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_height = frame.GetHeight();
  LONGS_EQUAL (height, local_height);
}

TEST (TfLiteFrame, FrameGetFormat) {
  r2i::RuntimeError error;
  r2i::ImageFormat local_format;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_format = frame.GetFormat();
  LONGS_EQUAL (format.GetId(), local_format.GetId());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
