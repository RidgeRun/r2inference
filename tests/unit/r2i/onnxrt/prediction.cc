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

#include <r2i/r2i.h>
#include <r2i/onnxrt/prediction.h>

#include <cstring>
#include <fstream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#define INPUTS 3

bool model_sucess = true;

TEST_GROUP (OnnxrtPrediction) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::onnxrt::Prediction> prediction =
    std::make_shared<r2i::onnxrt::Prediction>();
  float matrix[INPUTS] = {0.2, 0.4, 0.6};
  float *tensordata = nullptr;

  void setup () {
    tensordata = nullptr;
  }

  void teardown () {
  }
};

TEST (OnnxrtPrediction, SetTensorSuccess) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (OnnxrtPrediction, SetTensorNullValue) {
  r2i::RuntimeError error;

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (OnnxrtPrediction, SetTensorZeroSize) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->SetTensorValues(tensordata, 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (OnnxrtPrediction, Prediction) {
  r2i::RuntimeError error;
  tensordata = &matrix[0];
  double result = 0;

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  result = prediction->At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (OnnxrtPrediction, PredictionNoTensorValues) {
  r2i::RuntimeError error;

  prediction->At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (OnnxrtPrediction, PredictionNonExistentIndex) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  prediction->At (5, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (OnnxrtPrediction, GetResultSize) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  int output_size = prediction->GetResultSize ();
  LONGS_EQUAL (output_size, INPUTS * sizeof(float));
}

TEST (OnnxrtPrediction, GetResultData) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->SetTensorValues(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  void *internal_ptr = prediction->GetResultData ();
  MEMCMP_EQUAL (tensordata, internal_ptr, INPUTS * sizeof(float));
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
