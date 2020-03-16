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

#include <memory>
#include <r2i/r2i.h>
#include <r2i/tensorrt/model.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

bool infer;

#include "mockcudaengine.cc"
#include "mockruntime.cc"

#define GOOD_FILENAME __FILE__

TEST_GROUP (TensorRTModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Model model;

  void setup () {
    fail_runtime = false;
    bad_cached_engine = false;
    model = r2i::tensorrt::Model();
  }

  void teardown () {
  }
};

TEST (TensorRTModel, Start) {
  error = model.Start (GOOD_FILENAME);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, DoubleStart) {
  error = model.Start (GOOD_FILENAME);
  error = model.Start (GOOD_FILENAME);

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorRTModel, StartEmptyName) {
  error = model.Start ("");

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTModel, IncorrectRuntime) {
  fail_runtime = true;
  error = model.Start (GOOD_FILENAME);

  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

TEST (TensorRTModel, BadFile) {
  error = model.Start ("bad_file");

  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode());
}

TEST (TensorRTModel, BadChechedEngine) {
  bad_cached_engine = true;
  error = model.Start (GOOD_FILENAME);

  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
