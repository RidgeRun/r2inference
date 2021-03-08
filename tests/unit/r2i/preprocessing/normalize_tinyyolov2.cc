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

#include "r2i/preprocessing/normalize_tinyyolov2.h"

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define FRAME_WIDTH 2
#define FRAME_HEIGHT 2
#define FRAME_SIZE CHANNELS * FRAME_WIDTH * FRAME_HEIGHT
#define CHANNELS  3
#define TOLERANCE 0.000001

static const float reference_matrix_tinyyolov2[FRAME_SIZE] = {
  0.000000, 0.003922, 0.007843, 0.011765, 0.015686, 0.019608,
  0.023529, 0.027451, 0.031373, 0.035294, 0.039216, 0.043137
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
}

/* Override constructor to use smaller matrices for testing */
namespace r2i {
NormalizeTinyyoloV2::NormalizeTinyyoloV2 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};
}

TEST_GROUP(NormalizeTinyyoloV2) {
  r2i::RuntimeError error;
  /* Preprocessing modules */
  std::shared_ptr<r2i::IPreprocessing> preprocessing_tinyyolov2 =
    std::make_shared<r2i::NormalizeTinyyoloV2>();

  std::shared_ptr<unsigned char> dummy_frame_data;
  std::shared_ptr<mock::Frame> in_frame = std::make_shared<mock::Frame>();

  /* Output frames */
  std::shared_ptr<mock::Frame> out_frame_tinyyolov2 =
    std::make_shared<mock::Frame>();

  void setup() {
    error.Clean();

    /* Fill frames with dummy data */
    unsigned char value = 0;
    dummy_frame_data = std::shared_ptr<unsigned char>(new unsigned char[FRAME_SIZE],
                       std::default_delete<const unsigned char[]>());

    unsigned char *in_data = dummy_frame_data.get();

    for (unsigned int i = 0; i < FRAME_SIZE; i++) {
      in_data[i] = value;
      value++;
    }

    error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

    std::shared_ptr<float> out_data_tinyyolov2 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());

    error = out_frame_tinyyolov2->Configure (out_data_tinyyolov2.get(), FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);
  }
};

TEST(NormalizeTinyyoloV2, ApplySuccess) {
  error = preprocessing_tinyyolov2->Apply(in_frame, out_frame_tinyyolov2);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  /* Check output values are the expected ones */
  for (unsigned int i = 0; i < FRAME_SIZE; i++) {
    DOUBLES_EQUAL( reference_matrix_tinyyolov2[i],
                   static_cast<float *>(out_frame_tinyyolov2->GetData())[i], TOLERANCE);
  }

}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
