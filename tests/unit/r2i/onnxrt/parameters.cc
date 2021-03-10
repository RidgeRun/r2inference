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
#include <r2i/onnxrt/model.h>
#include <r2i/onnxrt/parameters.h>

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
  virtual r2i::RuntimeError Predict (std::shared_ptr<r2i::IFrame> in_frame,
                                     std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) { return r2i::RuntimeError(); }
};
}

namespace r2i {
namespace onnxrt {

Engine::Engine ()  {
  this->state = State::STOPPED;
}
RuntimeError Engine::Start () {
  RuntimeError error;
  this->state = State::STARTED;
  return error;
}
}
}

TEST_GROUP (OnnxrtParameters) {
};

TEST (OnnxrtParameters, ConfigureIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new mock::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (OnnxrtParameters, ConfigureIncompatibleModel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new mock::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (OnnxrtParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtParameters, ConfigureNullModel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);

  error = parameters.Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtParameters, ConfigureSuccess) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtParameters, SetAndGetModel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IModel> internalModel = parameters.GetModel();

  POINTERS_EQUAL(internalModel.get(), model.get());
}

TEST (OnnxrtParameters, SetAndGetEngine) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters.GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (OnnxrtParameters, SetUndefinedIntegerParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  value = 0;
  error = parameters.Set("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, GetUndefinedIntegerParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, SetUndefinedStringParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::string value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  value = "test";
  error = parameters.Set("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, GetUndefinedStringParameter) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::string value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::onnxrt::Model);

  error = parameters.Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("undefined-parameter", value);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, SetAndGetLogId) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
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

TEST (OnnxrtParameters, SetAndGetLoggingLevel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
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

TEST (OnnxrtParameters, SetAndGetIntraNumThreads) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
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

TEST (OnnxrtParameters, SetAndGetGraphOptLevel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
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

TEST (OnnxrtParameters, SetLogIdAtWrongEngineState) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::string in_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::onnxrt::Engine);
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

TEST (OnnxrtParameters, SetLoggingLevelAtWrongEngineState) {
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
  error = parameters.Set("logging-level", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtParameters, SetIntraNumThreadsAtWrongEngineState) {
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

TEST (OnnxrtParameters, SetGraphOptLevelAtWrongEngineState) {
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
  in_value = 2;
  error = parameters.Set("graph-optimization-level", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtParameters, SetWrongType) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int value = 0;

  error = parameters.Set("log-id", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, GetWrongType) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  int value = 0;

  error = parameters.Get("log-id", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtParameters, GetList) {
  r2i::RuntimeError error;
  r2i::onnxrt::Parameters parameters;
  std::vector<r2i::ParameterMeta> desc;

  error = parameters.List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
