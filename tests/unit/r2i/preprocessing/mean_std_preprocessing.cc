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

#include <r2i/preprocessing/mean_std_preprocessing.h>

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define FRAME_WIDTH 224
#define FRAME_HEIGHT 224
#define UNSUPPORTED_FRAME_WIDTH -1
#define UNSUPPORTED_FRAME_HEIGHT -1
#define CHANNELS  3

r2i::ImageFormat image_format_rgb (r2i::ImageFormat::Id::RGB);
std::vector<float> dummy_frame_data(CHANNELS * FRAME_WIDTH * FRAME_HEIGHT);

namespace mock {
class Frame : public r2i::IFrame {
 public:
  Frame () {}
  r2i::RuntimeError Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format) {

    for (unsigned int i; i < dummy_frame_data.size()-1; i++) {
	  dummy_frame_data.at(i) = i;
	}
   
    return r2i::RuntimeError();
  }

  void *GetData () {
    return dummy_frame_data.data();
  }
  
  int GetWidth () {
    return FRAME_WIDTH;
  }
  
  int GetHeight () {
    return FRAME_HEIGHT;
  }

  r2i::ImageFormat GetFormat () {
    return image_format_rgb;
  }

  r2i::DataType GetDataType () {
    return r2i::DataType::Id::FLOAT;
  }

 private:
  float *frame_data;
  int frame_width;
  int frame_height;
  r2i::ImageFormat frame_format;
};
}

TEST_GROUP(MeanStd) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPreprocessing> preprocessing = std::make_shared<r2i::MeanStdPreprocessing>();
  std::shared_ptr<mock::Frame> in_frame = std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame = std::make_shared<mock::Frame>();

  void setup() {
    error.Clean();
    error = in_frame->Configure(dummy_frame_data.data(), FRAME_WIDTH, FRAME_HEIGHT, image_format_rgb.GetId());
  }
};

TEST(MeanStd, ApplySuccess) {
  error = preprocessing->Apply(in_frame, out_frame, FRAME_WIDTH, FRAME_HEIGHT,
                               r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST(MeanStd, UnsupportedHeight) {
  error = preprocessing->Apply(in_frame, out_frame, FRAME_WIDTH, UNSUPPORTED_FRAME_HEIGHT,
                               r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(MeanStd, UnsupportedWidth) {
  error = preprocessing->Apply(in_frame, out_frame, FRAME_WIDTH, UNSUPPORTED_FRAME_WIDTH,
                               r2i::ImageFormat::Id::RGB);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(MeanStd, UnsupportedFormatId) {
  error = preprocessing->Apply(in_frame, out_frame, FRAME_WIDTH, UNSUPPORTED_FRAME_WIDTH,
                               r2i::ImageFormat::Id::BGR);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
