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
#include <r2i/tflite/engine.h>
#include <r2i/tflite/frame.h>
#include <r2i/tflite/prediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

void *dummy = nullptr;

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};

namespace r2i {
namespace tflite {

Model::Model () {}
r2i::RuntimeError Model::Start (const std::string &name) { return RuntimeError(); }
RuntimeError Model::Set (std::shared_ptr<::tflite::FlatBufferModel> tfltmodel) {
  this->tflite_model = tfltmodel;
  return r2i::RuntimeError();
};

std::shared_ptr<::tflite::FlatBufferModel> Model::GetTfliteModel () {
  std::unique_ptr<::tflite::FlatBufferModel> tflite_model;
  ::tflite::ErrorReporter *error_reporter = ::tflite::DefaultErrorReporter();
  std::string  path = "resources/squeezenet.tflite";
  tflite_model = ::tflite::FlatBufferModel::BuildFromFile(path.c_str(),
                 error_reporter);

  delete error_reporter;
  std::shared_ptr<::tflite::FlatBufferModel> tflite_model_shared{std::move(tflite_model)};
  return tflite_model_shared;
}

Frame::Frame () {}
void *Frame::GetData () {
  return dummy;
}
int Frame::GetWidth () { return 224; }
int Frame::GetHeight () { return 224; }

Prediction::Prediction () {}
double Prediction::At (unsigned int index,  r2i::RuntimeError &error) { return 0.0; }
void *Prediction::GetResultData () { return nullptr; }
unsigned int Prediction::GetResultSize () { return 0; }
}
}

TEST_GROUP (TfLiteEngine) {
  r2i::tflite::Engine engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;

  void setup () {
    model = std::make_shared<r2i::tflite::Model> ();
    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::tflite::Frame> ();
    dummy = malloc(sizeof(float));
  }

  void teardown () {
    free(dummy);
    dummy = nullptr;
  }
};

TEST (TfLiteEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TfLiteEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (TfLiteEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (TfLiteEngine, StartEngine) {

  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TfLiteEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TfLiteEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (TfLiteEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

TEST (TfLiteEngine, StartStopEngineTwice) {
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

TEST (TfLiteEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TfLiteEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TfLiteEngine, PredictEngine) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

int main (int ac, char **av) {
  /* This module detects fake leaks since the TF_Tensor couldn't
     be mocked since it's directly used by the predict module */
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
  return CommandLineTestRunner::RunAllTests (ac, av);
}
