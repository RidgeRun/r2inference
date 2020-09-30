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
#include <r2i/onnxrt/model.h>
#include <r2i/onnxrt_openvino/engine.h>
#include <r2i/onnxrt_openvino/parameters.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

namespace mock {
class Model : public r2i::IModel {
 public:
  Model () {}
  r2i::RuntimeError Start (const std::string &name) {
    return r2i::RuntimeError();
  }
};

class Engine : public r2i::IEngine {
 public:
  r2i::RuntimeError Start () {
    return r2i::RuntimeError();
  };
  r2i::RuntimeError Stop () {
    return r2i::RuntimeError();
  };
  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) {
    return r2i::RuntimeError();
  };
  virtual std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) { return nullptr; }
  virtual RuntimeError Predict (std::shared_ptr<r2i::IFrame> in_frame,
                                std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) { return r2i::RuntimeError(); }
};
}

namespace r2i {
namespace onnxrt {

Engine::Engine ()  {}
RuntimeError Engine::Start () {
  RuntimeError error;
  this->state = State::STARTED;
  return error;
}
}
}

TEST_GROUP (OnnxrtOpenVinoParameters) {
};

TEST (OnnxrtOpenVinoParameters, SettHardwareIdWithIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = "test";
  error = parameters.Set("hardware-id", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, GettHardwareIdWithIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  error = parameters.Get("hardware-id", out_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetHardwareIdAtWrongEngineState) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = engine->Start();

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = "test";
  error = parameters.Set("hardware-id", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetAndGetHardwareId) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = "test";
  error = parameters.Set("hardware-id", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("hardware-id", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
