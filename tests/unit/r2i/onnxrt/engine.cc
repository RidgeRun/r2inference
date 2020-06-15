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
#include <r2i/onnxrt/engine.h>

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};

TEST_GROUP (OnnxrtEngine) {
  r2i::onnxrt::Engine engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;

  void setup () {
    model = std::make_shared<r2i::onnxrt::Model> ();
    inc_model = std::make_shared<MockModel> ();
  }
};

TEST (OnnxrtEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (OnnxrtEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, StartStopEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

int main (int ac, char **av) {
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
  return CommandLineTestRunner::RunAllTests (ac, av);
}
