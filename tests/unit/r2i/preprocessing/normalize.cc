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

#include "r2i/preprocessing/normalize.h"
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define FRAME_WIDTH 2
#define FRAME_HEIGHT 2
#define FRAME_SIZE CHANNELS * FRAME_WIDTH * FRAME_HEIGHT
#define MEAN 5.5
#define STD_DEV 3.45205253
#define STD_DEV_ZERO 0.0
#define UNSUPPORTED_FRAME_WIDTH 1
#define UNSUPPORTED_FRAME_HEIGHT 1
#define CHANNELS  3
#define TOLERANCE 0.000001

static const float reference_matrix[FRAME_SIZE] = {
  -1.593255, -1.303572, -1.013890, -0.724207, -0.434524, -0.144841,
    0.144841, 0.434524, 0.724207, 1.013890, 1.303572, 1.593255
  };

namespace mock {
class Frame : public r2i::IFrame {
 public:
  Frame () {}
  r2i::RuntimeError Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format,
                               r2i::DataType::Id datatype_id) {

    r2i::RuntimeError error;
    r2i::ImageFormat imageformat (format);
    r2i::DataType datatype (datatype_id);

    if (nullptr == in_data) {
      error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
                 "Received a NULL data pointer");
      return error;
    }
    if (width <= 0) {
      error.Set (r2i::RuntimeError::Code::WRONG_API_USAGE,
                 "Received an invalid image width");
      return error;
    }
    if (height <= 0) {
      error.Set (r2i::RuntimeError::Code::WRONG_API_USAGE,
                 "Received an invalid image height");
      return error;
    }

    this->frame_data = static_cast<float *>(in_data);
    this->frame_width = width;
    this->frame_height = height;
    this->frame_format = imageformat;
    this->datatype = datatype;

    return error;
  }

  void *GetData () {
    return this->frame_data;
  }

  int GetWidth () {
    return this->frame_width;
  }

  int GetHeight () {
    return this->frame_height;
  }

  r2i::ImageFormat GetFormat () {
    return this->frame_format;
  }

  r2i::DataType GetDataType () {
    return this->datatype;
  }

 private:
  float *frame_data = nullptr;
  int frame_width;
  int frame_height;
  r2i::ImageFormat frame_format;
  r2i::DataType datatype;
};

class NormalizeMock : public r2i::Normalize {
 public:
  NormalizeMock () {
    this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
  }

  r2i::RuntimeError SetNormalizationParameters (unsigned char *
      frame_data, int width, int height, int channels) override {
    this->mean_red = MEAN;
    this->mean_green = MEAN;
    this->mean_blue = MEAN;
    this->std_dev_red = STD_DEV;
    this->std_dev_green = STD_DEV;
    this->std_dev_blue = STD_DEV;
    return r2i::RuntimeError();
  }
};

class InvalidNormalizeMock : public r2i::Normalize {
 public:
  InvalidNormalizeMock () {
    this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
  }

  r2i::RuntimeError SetNormalizationParameters (unsigned char *
      frame_data, int width, int height, int channels) override {
    this->mean_red = MEAN;
    this->mean_green = MEAN;
    this->mean_blue = MEAN;
    this->std_dev_red = STD_DEV_ZERO;
    this->std_dev_green = STD_DEV_ZERO;
    this->std_dev_blue = STD_DEV_ZERO;
    return r2i::RuntimeError();
  }

};

}

TEST_GROUP(Normalize) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPreprocessing> preprocessing =
    std::make_shared<mock::NormalizeMock>();
  std::shared_ptr<unsigned char> dummy_frame_data;
  std::shared_ptr<mock::Frame> in_frame = std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame = std::make_shared<mock::Frame>();
  unsigned char *in_data;
  unsigned char value;

  void setup() {
    error.Clean();
    /* Fill frames with dummy data */
    value = 0;
    dummy_frame_data = std::shared_ptr<unsigned char>(new unsigned char[FRAME_SIZE],
                       std::default_delete<const unsigned char[]>());

    in_data = dummy_frame_data.get();

    for (unsigned int i = 0; i < FRAME_SIZE; i++) {
      in_data[i] = value;
      value++;
    }

  }
};

TEST(Normalize, ApplySuccess) {
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  /* Check output values are the expected ones */
  for (unsigned int i = 0; i < FRAME_SIZE; i++) {
    DOUBLES_EQUAL( reference_matrix[i],
                   static_cast<float *>(out_frame->GetData())[i], TOLERANCE);
  }

}

TEST(Normalize, ZeroStdDev) {
  std::shared_ptr<r2i::IPreprocessing> invalid_preprocessing =
    std::make_shared<mock::InvalidNormalizeMock>();

  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  error = invalid_preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(Normalize, UnsupportedWidth) {
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[UNSUPPORTED_FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), UNSUPPORTED_FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, UnsupportedHeight) {
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * UNSUPPORTED_FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                UNSUPPORTED_FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, UnsupportedInputFormatId) {
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::BGR, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);
  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, UnsupportedOutputFormatId) {
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::BGR, r2i::DataType::Id::FLOAT);
  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, NullInputFrame) {
  in_frame = nullptr;
  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(Normalize, NullOutputFrame) {
  out_frame = nullptr;
  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(Normalize, NullInputFrameData) {
  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::BGR, r2i::DataType::Id::FLOAT);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(Normalize, NullOutputFrameData) {
  dummy_frame_data = std::shared_ptr<unsigned char>(new unsigned char[FRAME_SIZE],
                     std::default_delete<const unsigned char[]>());
  unsigned char *in_data = dummy_frame_data.get();
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
