/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/prediction.h>
#include <fstream>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (NcsdkPrediction) {
  r2i::ncsdk::Prediction prediction;
  float *data;
  float matrix[3] = {0.2, 0.4, 0.6};

  void setup () {
    data = matrix;
  }

  void teardown () {
  }
};

TEST (NcsdkPrediction, Prediction) {
  r2i::RuntimeError error;
  double result = 0;
  error = prediction.SetResult (data, 3);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  result = prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (NcsdkPrediction, PredictionNoData) {
  r2i::RuntimeError error;
  prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (NcsdkPrediction, PredictionNonExistentIndex) {
  r2i::RuntimeError error;
  prediction.At (4, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
