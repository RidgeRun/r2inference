/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <memory>
#include <r2i/r2i.h>
#include <r2i/ncsdk/frame.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#define SIZE_TEST 100
#define WIDTH_TEST 250
#define HEIGHT_TEST 250
#define NEGATIVE_TEST -1
#define INVALID_FORMAT 100

TEST_GROUP (NcsdkFrame) {
  r2i::RuntimeError error;
  r2i::ImageFormat format;
  r2i::ncsdk::Frame frame;

  void setup () {
  }

  void teardown () {
  }
};

TEST (NcsdkFrame, SetZeroWidth) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, 0, HEIGHT_TEST, r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetZeroHeight) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, 0, r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetNegativeWidth) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, NEGATIVE_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetNegativeHeight) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, NEGATIVE_TEST,
                           r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetGetWidth) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  auto width = frame.GetWidth ();
  LONGS_EQUAL (WIDTH_TEST, width);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkFrame, SetGetHeight) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  auto height = frame.GetHeight ();
  LONGS_EQUAL (WIDTH_TEST, height);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkFrame, SetNullData) {
  error = frame.Configure (nullptr, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (NcsdkFrame, SetGetData) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  float *data = static_cast<float *>(frame.GetData ());
  POINTERS_EQUAL (setdata, data);
}

TEST (NcsdkFrame, FormatCheckId) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  format = frame.GetFormat();
  LONGS_EQUAL (format.GetId(), r2i::ImageFormat::Id::RGB);
}

TEST (NcsdkFrame, FormatCheckPlanes) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  format = frame.GetFormat();
  LONGS_EQUAL (format.GetNumPlanes(), 3);
}

TEST (NcsdkFrame, FormatCheckDescription) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::RGB);
  format = frame.GetFormat();
  CHECK ("RGB" == format.GetDescription());
}

TEST (NcsdkFrame, InvalidFormatGetDescription) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::UNKNOWN_FORMAT);
  format = frame.GetFormat();
  CHECK ("Unknown format" == format.GetDescription());
}

TEST (NcsdkFrame, InvalidFormatGetNumPlanes) {
  float *setdata = (float *) malloc(SIZE_TEST);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Id::UNKNOWN_FORMAT);
  format = frame.GetFormat();
  LONGS_EQUAL (format.GetNumPlanes(), 0);
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
