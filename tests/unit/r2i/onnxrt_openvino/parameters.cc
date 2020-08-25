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
#include <r2i/onnxrt/engine.h>
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
};
}

namespace r2i {

RuntimeError r2i::onnxrt::Engine::Start () {
  RuntimeError error;
  this->state = State::STARTED;
  return error;
}

namespace onnxrt_openvino {

Engine::Engine ()  { }

}
}

TEST_GROUP (OnnxrtOpenVinoParameters) {
};

TEST (OnnxrtOpenVinoParameters, ConfigureIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new mock::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, ConfigureIncompatibleModel) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new mock::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, ConfigureNullModel) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);

  error = parameters.Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, ConfigureSuccess) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetAndGetEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters.GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (OnnxrtOpenVinoParameters, SetUndefinedIntegerParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  value = 0;
  error = parameters.Set("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, GetUndefinedIntegerParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetUndefinedStringParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  value = "test";
  error = parameters.Set("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, GetUndefinedStringParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  std::string value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetAndGetLogId) {
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
  error = parameters.Set("log-id", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("log-id", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (OnnxrtOpenVinoParameters, SetAndGetLoggingLevel) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 0;
  error = parameters.Set("logging-level", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("logging-level", out_value);

  LONGS_EQUAL(in_value, out_value);
}

TEST (OnnxrtOpenVinoParameters, SetAndGetIntraNumThreads) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 1;
  error = parameters.Set("intra-num-threads", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("intra-num-threads", out_value);

  LONGS_EQUAL(in_value, out_value);
}

TEST (OnnxrtOpenVinoParameters, SetAndGetGraphOptLevel) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 1;
  error = parameters.Set("graph-optimization-level", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("graph-optimization-level", out_value);

  LONGS_EQUAL(in_value, out_value);
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

TEST (OnnxrtOpenVinoParameters, SetLogIdAtWrongEngineState) {
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
  error = parameters.Set("log-id", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetLoggingLevelAtWrongEngineState) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = engine->Start();

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 0;
  error = parameters.Set("logging-level", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetIntraNumThreadsAtWrongEngineState) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = engine->Start();

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 0;
  error = parameters.Set("intra-num-threads", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, SetGraphOptLevelAtWrongEngineState) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt_openvino::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = engine->Start();

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Value different from the default */
  in_value = 2;
  error = parameters.Set("graph-optimization-level", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
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

TEST (OnnxrtOpenVinoParameters, SetWrongType) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int value = 0;

  error = parameters.Set("log-id", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, GetWrongType) {
  r2i::RuntimeError error;
  r2i::onnxrt_openvino::Parameters parameters;
  int value = 0;

  error = parameters.Get("log-id", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtOpenVinoParameters, GetList) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::vector<r2i::ParameterMeta> desc;

  error = parameters.List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
