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
#include <r2i/tensorrt/engine.h>
#include <r2i/tensorrt/model.h>
#include <r2i/tensorrt/frame.h>
#include <r2i/tensorrt/prediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#include "mockcudaengine.cc"

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {
    r2i::RuntimeError error; return error;
  }
};

void IExecutionContextDeleter (nvinfer1::IExecutionContext *p) {
  if (p)
    p->destroy ();
}

TEST_GROUP (TensorRTEngine) {
  r2i::tensorrt::Engine engine;
  std::shared_ptr<r2i::tensorrt::Model> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;
  r2i::RuntimeError error;

  std::shared_ptr<nvinfer1::IExecutionContext> buffer;

  void setup () {
    error.Clean();

    buffer = std::shared_ptr<nvinfer1::IExecutionContext> (new
             nvinfer1::MockExecutionContext, IExecutionContextDeleter);

    model = std::make_shared<r2i::tensorrt::Model> ();
    model->Set (buffer);

    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::tensorrt::Frame> ();
  }
};

TEST (TensorRTEngine, SetModel) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, SetModelNull) {
  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (TensorRTEngine, SetModelInvalid) {
  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTEngine, StartEngine) {
  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, StopEngine) {
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, PredictExecutionError) {
  std::shared_ptr<r2i::IPrediction> prediction;
  execute_error = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());

}

TEST (TensorRTEngine, PredictEngine) {
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
