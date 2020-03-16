/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <r2i/tensorrt/frame.h>
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
void *dummyprt = nullptr;


TEST_GROUP (TensorRTFrame) {
  r2i::tensorrt::Frame frame;
  int width = FRAME_WIDTH;
  int height = FRAME_HEIGHT;
  float data[4] = {0.1, 0.2, 0.3, 0.4};
  r2i::ImageFormat format;

  void setup () {
    frame = r2i::tensorrt::Frame();
    format = r2i::ImageFormat(r2i::ImageFormat::Id::RGB);
    dummyprt = malloc(1);
  }

  void teardown () {
    free(dummyprt);
  }
};

TEST (TensorRTFrame, FrameConfigure) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNullData) {
  r2i::RuntimeError error;

  error = frame.Configure(nullptr, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNegativeWidth) {
  r2i::RuntimeError error;

  error = frame.Configure(data, -1, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNegativeHeight) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, -1, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameGetData) {
  r2i::RuntimeError error;
  void *local_data;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_data = frame.GetData();
  POINTERS_EQUAL (data, local_data);
}

TEST (TensorRTFrame, FrameGetWidth) {
  r2i::RuntimeError error;
  int local_width;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_width = frame.GetWidth();
  LONGS_EQUAL (width, local_width);
}

TEST (TensorRTFrame, FrameGetHeight) {
  r2i::RuntimeError error;
  int local_height;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_height = frame.GetHeight();
  LONGS_EQUAL (height, local_height);
}

TEST (TensorRTFrame, FrameGetFormat) {
  r2i::RuntimeError error;
  r2i::ImageFormat local_format;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_format = frame.GetFormat();
  LONGS_EQUAL (format.GetId(), local_format.GetId());
}

TEST (TensorRTFrame, FrameGetTensor) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTFrame, FrameGetTensorNullGraph) {
  r2i::RuntimeError error;
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(nullptr, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorRTFrame, FrameGetTensorNullOperation) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, nullptr, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorRTFrame, FrameGetTensorMultipleBatches) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  multiple_batches = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  /* R2I will set the tensor size to 1 to allow frame by frame processing */
  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  multiple_batches = false;
}

TEST (TensorRTFrame, FrameGetTensorUnsupportedWidth) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_width = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_width = false;
}

TEST (TensorRTFrame, FrameGetTensorUnsupportedHeight) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_height = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_height = false;
}

TEST (TensorRTFrame, FrameGetTensorIncorrectChannels) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_channels = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_channels = false;
}

TEST (TensorRTFrame, FrameGetTensorNewTensorError) {
  r2i::RuntimeError error;
  // std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  // TF_Operation *operation = (TF_Operation *)
  //                           data; /* Used to avoid passing a nullptr */
  // std::shared_ptr<TF_Tensor> tensor(nullptr);

  new_tensor_error = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  // tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode());

  new_tensor_error = false;
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
