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

#include <r2i/predictiondecorator.h>

#include <cstring>
#include <fstream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#define INPUTS 3

bool model_sucess = true;

TEST_GROUP (PredictionDecorator) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::Prediction> prediction =
    std::make_shared<r2i::Prediction>();
  std::shared_ptr<r2i::PredictionDecorator> decorator =
    std::make_shared<r2i::PredictionDecorator>(prediction);
  float matrix[INPUTS] = {0.2, 0.4, 0.6};
  float *tensordata = nullptr;

  void setup () {
    tensordata = nullptr;
  }

  void teardown () {
  }
};

TEST (PredictionDecorator, SetTensorSuccess) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = decorator->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (PredictionDecorator, SetTensorNullValue) {
  r2i::RuntimeError error;

  error = decorator->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (PredictionDecorator, SetTensorZeroSize) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = decorator->AddResult(tensordata, 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (PredictionDecorator, Prediction) {
  r2i::RuntimeError error;
  tensordata = &matrix[0];
  double result = 0;

  error = prediction->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  result = decorator->At (0, 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (PredictionDecorator, PredictionNoTensorValues) {
  r2i::RuntimeError error;

  decorator->At (0, 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR,
               error.GetCode());
}

TEST (PredictionDecorator, PredictionNonExistentIndex) {
  r2i::RuntimeError error;

  tensordata = &matrix[0];

  error = prediction->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  decorator->At (0, 5, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

TEST (PredictionDecorator, ReplaceExistingPrediction) {
  r2i::RuntimeError error;
  tensordata = &matrix[0];
  unsigned int output_index = 0;

  error = prediction->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = decorator->InsertResult(output_index, tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (PredictionDecorator, ReplaceExistingPredictionWithWrongIndex) {
  r2i::RuntimeError error;
  tensordata = &matrix[0];
  unsigned int output_index = 2;

  error = prediction->AddResult(tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = decorator->InsertResult(output_index, tensordata, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
