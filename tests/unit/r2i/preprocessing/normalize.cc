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
#include "r2i/preprocessing/normalize_facenetv1.h"
#include "r2i/preprocessing/normalize_inceptionv1.h"
#include "r2i/preprocessing/normalize_inceptionv3.h"
#include "r2i/preprocessing/normalize_resnet50v1.h"
#include "r2i/preprocessing/normalize_tinyyolov2.h"
#include "r2i/preprocessing/normalize_tinyyolov3.h"

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define FRAME_WIDTH 2
#define FRAME_HEIGHT 2
#define FRAME_SIZE CHANNELS * FRAME_WIDTH * FRAME_HEIGHT
#define MEAN 0
#define STD_DEV 1
#define UNSUPPORTED_FRAME_WIDTH 1
#define UNSUPPORTED_FRAME_HEIGHT 1
#define CHANNELS  3
#define TOLERANCE 0.000001

static const float reference_matrix[FRAME_SIZE] = {
  1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.0
};

static const float reference_matrix_facenetv1[FRAME_SIZE] = {
  -1.550777, -1.218467, -0.886158, -0.886158, -0.553849, -0.221540,
    0.110770, 0.443079, 0.775388, 1.107698, 1.440007, 1.440007
  };

static const float reference_matrix_inception[FRAME_SIZE] = {
  -0.992188, -0.984375, -0.976562, -0.976562, -0.968750, -0.960938,
    -0.953125, -0.945312, -0.937500, -0.929688, -0.921875, -0.921875
  };

static const float reference_matrix_resnet50v1[FRAME_SIZE] = {
  -122.680000, -114.779999, -100.940002, -120.680000, -112.779999, -98.940002,
    -117.680000, -109.779999, -95.940002, -114.680000, -106.779999, -93.940002
  };

static const float reference_matrix_tinyyolov2[FRAME_SIZE] = {
  0.003922, 0.007843, 0.011765, 0.011765, 0.015686, 0.019608,
  0.023529, 0.027451, 0.031373, 0.035294, 0.039216, 0.039216
};

namespace mock {
class Frame : public r2i::IFrame {
 public:
  Frame () {}
  r2i::RuntimeError Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format) {

    r2i::RuntimeError error;
    r2i::ImageFormat imageformat (format);

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
    return r2i::DataType::Id::FLOAT;
  }

 private:
  float *frame_data = nullptr;
  int frame_width;
  int frame_height;
  r2i::ImageFormat frame_format;
};

class NormalizeMock : public r2i::Normalize {

 public:
  NormalizeMock () {
    this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
  }

 private:
  r2i::RuntimeError SetNormalizationParameters () {
    this->mean_red = MEAN;
    this->mean_green = MEAN;
    this->mean_blue = MEAN;
    this->std_dev_red = STD_DEV;
    this->std_dev_green = STD_DEV;
    this->std_dev_blue = STD_DEV;
    return r2i::RuntimeError();
  }

};
}

/* Override constructors to use smaller matrices for testing */
namespace r2i {

NormalizeFaceNetV1::NormalizeFaceNetV1 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

NormalizeInceptionV1::NormalizeInceptionV1 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

NormalizeInceptionV3::NormalizeInceptionV3 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

NormalizeResnet50V1::NormalizeResnet50V1 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

NormalizeTinyyoloV2::NormalizeTinyyoloV2 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

NormalizeTinyyoloV3::NormalizeTinyyoloV3 () {
  this->dimensions.push_back(std::tuple<int, int>(FRAME_WIDTH, FRAME_HEIGHT));
};

}

TEST_GROUP(Normalize) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPreprocessing> preprocessing =
    std::make_shared<mock::NormalizeMock>();
  std::shared_ptr<unsigned char> dummy_frame_data;
  std::shared_ptr<mock::Frame> in_frame = std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame = std::make_shared<mock::Frame>();

  void setup() {
    error.Clean();
  }
};

TEST(Normalize, ApplySuccess) {
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
                              r2i::ImageFormat::Id::RGB);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  /* Check output values are the expected ones */
  for (unsigned int i = 0; i < FRAME_SIZE; i++) {
    DOUBLES_EQUAL( reference_matrix[i],
                   static_cast<float *>(out_frame->GetData())[i], TOLERANCE);
  }

}

TEST(Normalize, UnsupportedWidth) {
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
                              r2i::ImageFormat::Id::RGB);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[UNSUPPORTED_FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), UNSUPPORTED_FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, UnsupportedHeight) {
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
                              r2i::ImageFormat::Id::RGB);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * UNSUPPORTED_FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                UNSUPPORTED_FRAME_HEIGHT,
                                r2i::ImageFormat::Id::RGB);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::MODULE_ERROR, error.GetCode());
}

TEST(Normalize, UnsupportedFormatId) {
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
                              r2i::ImageFormat::Id::RGB);

  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), FRAME_WIDTH,
                                FRAME_HEIGHT,
                                r2i::ImageFormat::Id::BGR);
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
                                r2i::ImageFormat::Id::BGR);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(Normalize, NullOutputFrameData) {
  dummy_frame_data = std::shared_ptr<unsigned char>(new unsigned char[FRAME_SIZE],
                     std::default_delete<const unsigned char[]>());
  unsigned char *in_data = dummy_frame_data.get();
  error = in_frame->Configure(in_data, FRAME_WIDTH, FRAME_HEIGHT,
                              r2i::ImageFormat::Id::RGB);

  error = preprocessing->Apply(in_frame, out_frame);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST_GROUP(NormalizeModels) {
  r2i::RuntimeError error;
  /* Preprocessing modules */
  std::shared_ptr<r2i::IPreprocessing> preprocessing_facenetv1 =
    std::make_shared<r2i::NormalizeFaceNetV1>();
  std::shared_ptr<r2i::IPreprocessing> preprocessing_inceptionv1 =
    std::make_shared<r2i::NormalizeInceptionV1>();
  std::shared_ptr<r2i::IPreprocessing> preprocessing_inceptionv3 =
    std::make_shared<r2i::NormalizeInceptionV3>();
  std::shared_ptr<r2i::IPreprocessing> preprocessing_resnet50v1 =
    std::make_shared<r2i::NormalizeResnet50V1>();
  std::shared_ptr<r2i::IPreprocessing> preprocessing_tinyyolov2 =
    std::make_shared<r2i::NormalizeTinyyoloV2>();
  std::shared_ptr<r2i::IPreprocessing> preprocessing_tinyyolov3 =
    std::make_shared<r2i::NormalizeTinyyoloV3>();

  std::shared_ptr<unsigned char> dummy_frame_data;
  std::shared_ptr<mock::Frame> in_frame = std::make_shared<mock::Frame>();

  /* Output frames */
  std::shared_ptr<mock::Frame> out_frame_facenetv1 =
    std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame_inceptionv1 =
    std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame_inceptionv3 =
    std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame_resnet50v1 =
    std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame_tinyyolov2 =
    std::make_shared<mock::Frame>();
  std::shared_ptr<mock::Frame> out_frame_tinyyolov3 =
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
                                r2i::ImageFormat::Id::RGB);

    std::shared_ptr<float> out_data_facenetv1 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    std::shared_ptr<float> out_data_inceptionv1 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    std::shared_ptr<float> out_data_inceptionv3 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    std::shared_ptr<float> out_data_resnet50v1 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    std::shared_ptr<float> out_data_tinyyolov2 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    std::shared_ptr<float> out_data_tinyyolov3 = std::shared_ptr<float>
        (new float[FRAME_WIDTH * FRAME_HEIGHT * CHANNELS],
         std::default_delete<float[]>());
    error = out_frame_facenetv1->Configure (out_data_facenetv1.get(), FRAME_WIDTH,
                                            FRAME_HEIGHT,
                                            r2i::ImageFormat::Id::RGB);
    error = out_frame_inceptionv1->Configure (out_data_inceptionv1.get(),
            FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB);
    error = out_frame_inceptionv3->Configure (out_data_inceptionv3.get(),
            FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB);
    error = out_frame_resnet50v1->Configure (out_data_resnet50v1.get(), FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB);
    error = out_frame_tinyyolov2->Configure (out_data_tinyyolov2.get(), FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB);
    error = out_frame_tinyyolov3->Configure (out_data_tinyyolov3.get(), FRAME_WIDTH,
            FRAME_HEIGHT,
            r2i::ImageFormat::Id::RGB);

  }
};

TEST(NormalizeModels, ApplySuccess) {
  error = preprocessing_facenetv1->Apply(in_frame, out_frame_facenetv1);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  error = preprocessing_inceptionv1->Apply(in_frame, out_frame_inceptionv1);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  error = preprocessing_inceptionv3->Apply(in_frame, out_frame_inceptionv3);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  error = preprocessing_resnet50v1->Apply(in_frame, out_frame_resnet50v1);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  error = preprocessing_tinyyolov2->Apply(in_frame, out_frame_tinyyolov2);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  error = preprocessing_tinyyolov3->Apply(in_frame, out_frame_tinyyolov3);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());

  /* Check output values are the expected ones */
  for (unsigned int i = 0; i < FRAME_SIZE; i++) {
    DOUBLES_EQUAL( reference_matrix_facenetv1[i],
                   static_cast<float *>(out_frame_facenetv1->GetData())[i], TOLERANCE);
    DOUBLES_EQUAL( reference_matrix_inception[i],
                   static_cast<float *>(out_frame_inceptionv1->GetData())[i], TOLERANCE);
    DOUBLES_EQUAL( reference_matrix_inception[i],
                   static_cast<float *>(out_frame_inceptionv3->GetData())[i], TOLERANCE);
    DOUBLES_EQUAL( reference_matrix_resnet50v1[i],
                   static_cast<float *>(out_frame_resnet50v1->GetData())[i], TOLERANCE);
    DOUBLES_EQUAL( reference_matrix_tinyyolov2[i],
                   static_cast<float *>(out_frame_tinyyolov2->GetData())[i], TOLERANCE);
    DOUBLES_EQUAL( reference_matrix[i],
                   static_cast<float *>(out_frame_tinyyolov3->GetData())[i], TOLERANCE);
  }

}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
