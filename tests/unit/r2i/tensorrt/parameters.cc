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
#include <r2i/tensorrt/parameters.h>
#include <r2i/tensorrt/prediction.h>

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
namespace tensorrt {
Model::Model () { }

RuntimeError Model::Start (const std::string &name) { return RuntimeError(); }


Engine::Engine ()  { }
RuntimeError Engine::SetModel (std::shared_ptr<IModel> in_model) { return RuntimeError(); }
RuntimeError Engine::Start () { return RuntimeError(); }
RuntimeError Engine::Stop () { return RuntimeError(); }
std::shared_ptr<IPrediction> Engine::Predict (std::shared_ptr<IFrame> in_frame,
    RuntimeError &error) { return nullptr; }
Engine::~Engine () { }

}
}

TEST_GROUP (TensorrtParameters) {
};

TEST (TensorrtParameters, ConfigureIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new mock::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (TensorrtParameters, ConfigureIncompatibleModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new mock::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorrtParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorrtParameters, ConfigureNullModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);

  error = parameters.Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorrtParameters, ConfigureSuccess) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorrtParameters, SetAndGetModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IModel> internalModel = parameters.GetModel();

  POINTERS_EQUAL(internalModel.get(), model.get());
}

TEST (TensorrtParameters, SetAndGetEngine) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters.GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (TensorrtParameters, SetAndGetInputLayerName) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = "myinputlayer";
  error = parameters.Set("input-layer", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("input-layer", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorrtParameters, SetAndGetOutputLayerName) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = "myoutputlayer";
  error = parameters.Set("output-layer", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("output-layer", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorrtParameters, GetVersion) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::string value = "";

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("version", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  CHECK(! value.compare(value));
}

TEST (TensorrtParameters, GetMissingString) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::string value = "value";

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, GetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  int value;

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, SetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  int value = 0;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, SetMissingString) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::string value;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, SetWrongType) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, GetWrongType) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorrtParameters, GetList) {
  r2i::RuntimeError error;
  r2i::tensorrt::Parameters parameters;
  std::vector<r2i::ParameterMeta> desc;

  error = parameters.List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
