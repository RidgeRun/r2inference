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
#include <r2i/tensorflow/frame.h>
#include <fstream>
#include <cstring>
#include <memory>

#include <tensorflow/c/c_api.h>

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

int TF_GraphGetTensorNumDims(TF_Graph *graph, TF_Output output,
                             TF_Status *status) {
  return 4;
}
void TF_GraphGetTensorShape(TF_Graph *graph, TF_Output output, int64_t *dims,
                            int num_dims, TF_Status *status) {
  if (multiple_batches) {
    dims[0] = 3;
  } else {
    dims[0] = 1;
  }

  if (invalid_width) {
    dims[1] = -1;
  } else {
    dims[1] = FRAME_WIDTH;
  }

  if (invalid_height) {
    dims[2] = -1;
  } else {
    dims[2] = FRAME_HEIGHT;
  }

  if (invalid_channels) {
    dims[3] = -1;
  } else {
    dims[3] = 3;
  }

}
TF_DataType TF_OperationOutputType(TF_Output oper_out) {
  return TF_FLOAT;
}
TF_Graph *TF_NewGraph() { return (TF_Graph *) dummyprt; }
void TF_DeleteGraph(TF_Graph *g) {
}
TF_Tensor *TF_NewTensor(TF_DataType dtype, const int64_t *dims, int num_dims,
                        void *data, size_t len, void (*deallocator)(void *data, size_t len, void *arg),
                        void *deallocator_arg) {
  if (new_tensor_error) {
    return (TF_Tensor *) nullptr;
  } else {
    return (TF_Tensor *) dummyprt;
  }
}
void TF_DeleteTensor(TF_Tensor *g) {
}

TEST_GROUP (TensorflowFrame) {
  r2i::tensorflow::Frame frame;
  int width = FRAME_WIDTH;
  int height = FRAME_HEIGHT;
  float data[4] = {0.1, 0.2, 0.3, 0.4};
  r2i::ImageFormat format;

  void setup () {
    frame = r2i::tensorflow::Frame();
    format = r2i::ImageFormat(r2i::ImageFormat::Id::RGB);
    dummyprt = malloc(1);
  }

  void teardown () {
    free(dummyprt);
  }
};

TEST (TensorflowFrame, FrameConfigure) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowFrame, FrameConfigureNullData) {
  r2i::RuntimeError error;

  error = frame.Configure(nullptr, width, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorflowFrame, FrameConfigureNegativeWidth) {
  r2i::RuntimeError error;

  error = frame.Configure(data, -1, height, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorflowFrame, FrameConfigureNegativeHeight) {
  r2i::RuntimeError error;

  error = frame.Configure(data, width, -1, format.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorflowFrame, FrameGetData) {
  r2i::RuntimeError error;
  void *local_data;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_data = frame.GetData();
  POINTERS_EQUAL (data, local_data);
}

TEST (TensorflowFrame, FrameGetWidth) {
  r2i::RuntimeError error;
  int local_width;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_width = frame.GetWidth();
  LONGS_EQUAL (width, local_width);
}

TEST (TensorflowFrame, FrameGetHeight) {
  r2i::RuntimeError error;
  int local_height;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_height = frame.GetHeight();
  LONGS_EQUAL (height, local_height);
}

TEST (TensorflowFrame, FrameGetFormat) {
  r2i::RuntimeError error;
  r2i::ImageFormat local_format;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_format = frame.GetFormat();
  LONGS_EQUAL (format.GetId(), local_format.GetId());
}

TEST (TensorflowFrame, FrameGetTensor) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowFrame, FrameGetTensorNullGraph) {
  r2i::RuntimeError error;
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(nullptr, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowFrame, FrameGetTensorNullOperation) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, nullptr, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowFrame, FrameGetTensorMultipleBatches) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  multiple_batches = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  /* R2I will set the tensor size to 1 to allow frame by frame processing */
  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  multiple_batches = false;
}

TEST (TensorflowFrame, FrameGetTensorUnsupportedWidth) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_width = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_width = false;
}

TEST (TensorflowFrame, FrameGetTensorUnsupportedHeight) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_height = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_height = false;
}

TEST (TensorflowFrame, FrameGetTensorIncorrectChannels) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  invalid_channels = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());

  invalid_channels = false;
}

TEST (TensorflowFrame, FrameGetTensorNewTensorError) {
  r2i::RuntimeError error;
  std::shared_ptr<TF_Graph> pgraph(TF_NewGraph(), TF_DeleteGraph);
  TF_Operation *operation = (TF_Operation *)
                            data; /* Used to avoid passing a nullptr */
  std::shared_ptr<TF_Tensor> tensor(nullptr);

  new_tensor_error = true;

  error = frame.Configure(data, width, height, format.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  tensor = frame.GetTensor(pgraph, operation, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode());

  new_tensor_error = false;
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
