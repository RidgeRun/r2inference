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

#include <CppUTest/TestHarness.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/frame.h>
#include <memory>

#define SIZE_TEST 100
#define WIDTH_TEST 250
#define HEIGHT_TEST 250
#define NEGATIVE_TEST -1

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
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, 0, HEIGHT_TEST, r2i::ImageFormat::Code::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetZeroHeight) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, 0, r2i::ImageFormat::Code::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetNegativeWidth) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, NEGATIVE_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetNegativeHeight) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, NEGATIVE_TEST,
                           r2i::ImageFormat::Code::RGB);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkFrame, SetGetWidth) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  auto width = frame.GetWidth ();
  LONGS_EQUAL (WIDTH_TEST, width);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkFrame, SetGetHeight) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  auto height = frame.GetHeight ();
  LONGS_EQUAL (WIDTH_TEST, height);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkFrame, SetNullData) {
  error = frame.Configure (nullptr, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  auto data = frame.GetData ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (NcsdkFrame, SetGetData) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  auto data = frame.GetData ();
  POINTERS_EQUAL (setdata.get(), data.get());
}

TEST (NcsdkFrame, FormatCheckCode) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  format = frame.GetFormat();
  LONGS_EQUAL (format.GetCode(), r2i::ImageFormat::Code::RGB);
}

TEST (NcsdkFrame, FormatCheckPlanes) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  format = frame.GetFormat();
  LONGS_EQUAL (format.GetNumPlanes(), 3);
}

TEST (NcsdkFrame, FormatCheckDescription) {
  std::shared_ptr<void> setdata (malloc(SIZE_TEST), free);
  error = frame.Configure (setdata, WIDTH_TEST, HEIGHT_TEST,
                           r2i::ImageFormat::Code::RGB);
  format = frame.GetFormat();
  CHECK ("RGB" == format.GetDescription());
}
