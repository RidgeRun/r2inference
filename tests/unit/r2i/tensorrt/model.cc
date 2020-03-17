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

#define GOOD_FILENAME __FILE__

void IExecutionContextDeleter (nvinfer1::IExecutionContext *p) {
  if (p)
    p->destroy ();
}

TEST_GROUP (TensorRTModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Model model;
  std::shared_ptr<nvinfer1::IExecutionContext> buffer;

  void setup () {
    model = r2i::tensorrt::Model();
    buffer = std::shared_ptr<nvinfer1::IExecutionContext> (new
             nvinfer1::MockExecutionContext,
             IExecutionContextDeleter);
  }

  void teardown () {
  }
};

TEST (TensorRTModel, Start) {
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, SetSuccess) {
  error = model.Set (buffer);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, SetNullBuffer) {
  error = model.Set (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTModel, GetSuccess) {
  std::shared_ptr<nvinfer1::IExecutionContext> model_buffer =
    model.GetTRContext ();

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, GetNullBuffer) {
  std::shared_ptr<nvinfer1::IExecutionContext> model_buffer =
    model.GetTRContext ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
