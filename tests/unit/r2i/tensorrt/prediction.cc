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
#include <r2i/tensorrt/prediction.h>

#include <cstring>
#include <fstream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define INPUTS 3

bool cudaMemCpyError = false;

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src,
    size_t count, enum cudaMemcpyKind kind) {
  if (!cudaMemCpyError) {
    memcpy ( dst, src, count );
    return cudaSuccess;
  } else {
    return cudaErrorInvalidValue;
  }
}

TEST_GROUP (TensorRTPrediction) {
  r2i::RuntimeError error;

  r2i::tensorrt::Prediction prediction;
  std::shared_ptr<float[]> matrix = std::shared_ptr<float[]> (new float[INPUTS]);

  int64_t raw_input_dims[1] = {INPUTS};

  void setup () {
    error.Clean();
    cudaMemCpyError = false;

    matrix[0] = 0.2;
    matrix[1] = 0.4;
    matrix[2] = 0.6;
  }

  void teardown () {
  }
};

TEST (TensorRTPrediction, SetResultBufferSuccess) {
  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTPrediction, SetNullResultBuffer) {
  error = prediction.AddResult(nullptr, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTPrediction, Prediction) {
  double result;

  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  result = prediction.At (0, 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (TensorRTPrediction, PredictionGetResultSize) {
  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  unsigned int result_size = prediction.GetResultSize(0, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  LONGS_EQUAL (INPUTS, result_size);
}

TEST (TensorRTPrediction, PredictionNoTensor) {
  prediction.At (0, 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR,
               error.GetCode());
}

TEST (TensorRTPrediction, PredictionNonExistentIndex) {
  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  prediction.At (0, 5, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (TensorRTPrediction, PredictionCudaMemCpyError) {
  cudaMemCpyError = true;
  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (TensorRTPrediction, ReplaceExistingPrediction) {
  r2i::RuntimeError error;
  unsigned int output_index = 0;

  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = prediction.InsertResult(output_index, matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTPrediction, ReplaceExistingPredictionWithWrongIndex) {
  r2i::RuntimeError error;
  unsigned int output_index = 2;

  error = prediction.AddResult(matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = prediction.InsertResult(output_index, matrix.get(), INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
