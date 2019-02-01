/* Copyright (C) 2019 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <r2i/tensorflow/parameters.h>
#include <r2i/tensorflow/prediction.h>

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
namespace tensorflow {
Model::Model () { }

RuntimeError Model::Start (const std::string &name) { return RuntimeError(); }

RuntimeError Model::Load (std::shared_ptr<TF_Buffer> pbuffer) {
  this->buffer = pbuffer;
  return RuntimeError();
}

std::shared_ptr<TF_Graph> Model::GetGraph () {return this->graph;}
std::shared_ptr<TF_Buffer> Model::GetBuffer () {return this->buffer;}
TF_Operation *Model::GetInputOperation () { return nullptr; }
TF_Operation *Model::GetOutputOperation () { return nullptr; }
RuntimeError Model::SetInputLayerName (const std::string &name) {
  this->input_layer_name = name;
  return RuntimeError();
}
RuntimeError Model::SetOutputLayerName (const std::string &name) {
  this->output_layer_name = name;
  return RuntimeError();
}
const std::string Model::GetInputLayerName () { return this->input_layer_name; }
const std::string Model::GetOutputLayerName () { return this->output_layer_name; }

Engine::Engine ()  { }
RuntimeError Engine::SetModel (std::shared_ptr<IModel> in_model) { return RuntimeError(); }
RuntimeError Engine::Start () { return RuntimeError(); }
RuntimeError Engine::Stop () { return RuntimeError(); }
std::shared_ptr<IPrediction> Engine::Predict (std::shared_ptr<IFrame> in_frame,
    RuntimeError &error) { return nullptr; }
Engine::~Engine () { }

}
}


TEST_GROUP (TensorflowParameters) {
};

TEST (TensorflowParameters, ConfigureIncompatibleEngine) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new mock::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (TensorflowParameters, ConfigureIncompatibleModel) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new mock::Model);

  error = parameters.Configure(engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorflowParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorflowParameters, ConfigureNullModel) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);

  error = parameters.Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorflowParameters, ConfigureSuccess) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorflowParameters, SetAndGetModel) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IModel> internalModel = parameters.GetModel();

  POINTERS_EQUAL(internalModel.get(), model.get());
}

TEST (TensorflowParameters, SetAndGetEngine) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters.GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (TensorflowParameters, SetAndGetInputLayerName) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = "myinputlayer";
  error = parameters.Set("input-layer", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("input-layer", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorflowParameters, SetAndGetOutputLayerName) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::string in_value;
  std::string out_value;

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  in_value = "myoutputlayer";
  error = parameters.Set("output-layer", in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("output-layer", out_value);

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorflowParameters, GetVersion) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::string value = "";

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorflow::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorflow::Model);

  error = parameters.Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters.Get("version", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  CHECK(! value.compare(value));
}

TEST (TensorflowParameters, GetMissingString) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::string value = "value";

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, GetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  int value;

  error = parameters.Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, SetMissingInteger) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  int value = 0;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, SetMissingString) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::string value;

  error = parameters.Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, SetWrongType) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, GetWrongType) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  int value = 0;

  error = parameters.Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorflowParameters, GetList) {
  r2i::RuntimeError error;
  r2i::tensorflow::Parameters parameters;
  std::vector<r2i::ParameterMeta> desc;

  error = parameters.List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}


int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
