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
#include <r2i/tflite/parameters.h>
#include <r2i/tflite/engine.h>
#include <r2i/tflite/prediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

namespace mock {

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
namespace tflite {

Engine::Engine ()  { }
RuntimeError Engine::SetModel (std::shared_ptr<IModel> in_model) { return RuntimeError(); }
RuntimeError Engine::Start () { return RuntimeError(); }
RuntimeError Engine::Stop () { return RuntimeError(); }

RuntimeError Engine::SetNumberOfThreads (int name) {
  this->number_of_threads = name;
  return RuntimeError();
}
const int Engine::GetNumberOfThreads () { return this->number_of_threads; }

RuntimeError Engine::SetAllowFP16 (int name) {
  this->allow_fp16 = name;
  return RuntimeError();
}
const int Engine::GetAllowFP16 () { return this->allow_fp16; }

std::shared_ptr<IPrediction> Engine::Predict (std::shared_ptr<IFrame> in_frame,
    RuntimeError &error) { return nullptr; }
Engine::~Engine () { }

}
}

TEST_GROUP (TliteParameters) {
};

TEST (TliteParameters, ConfigureIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new mock::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (TliteParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;

  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TliteParameters, ConfigureNullModel) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tflite::Engine);

  error = parameters.Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TliteParameters, ConfigureSuccess) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tflite::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TliteParameters, SetAndGetEngine) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tflite::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters.GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (TliteParameters, SetAndGetNumberOfThreads) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tflite::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = 0;
  error = parameters.Set("number_of_threads", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("number_of_threads", out_value);

  CHECK_EQUAL(in_value, out_value);
}

TEST (TliteParameters, SetAndGetAllowFP16) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int in_value;
  int out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tflite::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tflite::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = 0;
  error = parameters.Set("allow_fp16", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("allow_fp16", out_value);

  CHECK_EQUAL(in_value, out_value);
}

TEST (TliteParameters, GetMissingString) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  std::string value = "value";

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, GetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int value;

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, SetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int value = 0;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, SetMissingString) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  std::string value;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, SetWrongType) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, GetWrongType) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TliteParameters, GetList) {
  r2i::RuntimeError error;
  r2i::tflite::Parameters parameters;
  std::vector<r2i::ParameterMeta> desc;

  error = parameters.List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
