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

#include <cuda.h>
#include <cuda_runtime.h>

#define FRAME_WIDTH  2
#define FRAME_HEIGHT  2

bool cudaMallocError = false;
bool cudaMemCpyError = false;
size_t cudaRequestedSize = 0;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMalloc(void **devPtr, size_t size) {
  if (!cudaMallocError) {
    cudaRequestedSize = size;
    return cudaSuccess;
  } else {
    return cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
    size_t count, enum cudaMemcpyKind kind) {
  if (!cudaMemCpyError)
    return cudaSuccess;
  else
    return cudaErrorInvalidValue;
}

TEST_GROUP (TensorRTFrame) {
  r2i::tensorrt::Frame frame;
  int width = FRAME_WIDTH;
  int height = FRAME_HEIGHT;
  float data[4] = {0.1, 0.2, 0.3, 0.4};
  r2i::ImageFormat format;
  r2i::DataType type;
  r2i::RuntimeError error;

  void setup () {
    error.Clean();
    frame = r2i::tensorrt::Frame();
    format = r2i::ImageFormat(r2i::ImageFormat::Id::RGB);
    type = r2i::DataType(r2i::DataType::Id::FLOAT);

    cudaMallocError = false;
    cudaMemCpyError = false;
    cudaRequestedSize = 0;
  }

  void teardown () {
  }
};

TEST (TensorRTFrame, FrameConfigure) {
  error = frame.Configure(data, width, height, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTFrame, FrameCheckSize) {
  error = frame.Configure(data, width, height, format.GetId(), type.GetId());

  LONGS_EQUAL (FRAME_WIDTH * FRAME_HEIGHT * format.GetNumPlanes() *
               type.GetBytesPerPixel(), cudaRequestedSize);
}

TEST (TensorRTFrame, FrameConfigureNullData) {
  error = frame.Configure(nullptr, width, height, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNullWidth) {
  error = frame.Configure(data, 0, height, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNegativeWidth) {
  error = frame.Configure(data, -1, height, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNullHeight) {
  error = frame.Configure(data, width, 0, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureNegativeHeight) {
  error = frame.Configure(data, width, -1, format.GetId(), type.GetId());

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameConfigureUnknownDataType) {
  error = frame.Configure(data, width, 0, format.GetId(),
                          r2i::DataType::Id::UNKNOWN_DATATYPE);

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTFrame, FrameGetWidth) {
  int local_width;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_width = frame.GetWidth();
  LONGS_EQUAL (width, local_width);
}

TEST (TensorRTFrame, FrameGetHeight) {
  int local_height;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_height = frame.GetHeight();
  LONGS_EQUAL (height, local_height);
}

TEST (TensorRTFrame, FrameGetFormat) {
  r2i::ImageFormat local_format;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_format = frame.GetFormat();
  LONGS_EQUAL (format.GetId(), local_format.GetId());
}

TEST (TensorRTFrame, FrameGetDataType) {
  r2i::DataType local_data_type;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  local_data_type = frame.GetDataType();
  LONGS_EQUAL (type.GetId(), local_data_type.GetId());
}

TEST (TensorRTFrame, FrameCudaMallocError) {
  cudaMallocError = true;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (TensorRTFrame, FrameCudaMemCpyError) {
  cudaMemCpyError = true;

  error = frame.Configure(data, width, height, format.GetId(), type.GetId());
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
