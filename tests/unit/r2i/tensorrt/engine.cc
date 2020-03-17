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
#include <r2i/tensorrt/frame.h>
#include <r2i/tensorrt/prediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};


namespace r2i {
namespace tensorrt {

Model::Model () {}
r2i::RuntimeError Model::Start (const std::string &name) { return RuntimeError(); }
r2i::RuntimeError Model::Load (std::shared_ptr<TF_Buffer> pbuffer) {
  this->buffer = pbuffer;
  return r2i::RuntimeError();
}

Frame::Frame () {}
RuntimeError Frame::Configure (void *in_data, int width, int height,
                               r2i::ImageFormat::Id format, r2i::DataType::Id type) { return RuntimeError(); }
void *Frame::GetData () { return nullptr; }
int Frame::GetWidth () { return 0; }
int Frame::GetHeight () { return 0; }
ImageFormat Frame::GetFormat () { return ImageFormat::Id::UNKNOWN_FORMAT; }

Prediction::Prediction () {}
double Prediction::At (unsigned int index,  r2i::RuntimeError &error) { return 0.0; }
void *Prediction::GetResultData () { return nullptr; }
unsigned int Prediction::GetResultSize () { return 0; }
}
}

TEST_GROUP (TensorRTEngine) {
  r2i::tensorrt::Engine engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;

  void setup () {
    model = std::make_shared<r2i::tensorrt::Model> ();

    auto tf_model = std::dynamic_pointer_cast<r2i::tensorrt::Model, r2i::IModel>
                    (model);

    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::tensorrt::Frame> ();
  }
};

TEST (TensorRTEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (TensorRTEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTEngine, StartEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (TensorRTEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

TEST (TensorRTEngine, StartStopEngineTwice) {
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

TEST (TensorRTEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TensorRTEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TensorRTEngine, PredictEngine) {
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
  return CommandLineTestRunner::RunAllTests (ac, av);
}
