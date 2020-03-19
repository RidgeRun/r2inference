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

#include "mockcudaengine.cc"

static void ICudaEngineDeleter (nvinfer1::ICudaEngine *p) {
  if (p)
    p->destroy ();
}

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

TEST_GROUP (TensorRTParameters) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::tensorrt::Parameters> parameters;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;

  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IEngine> engine;

  void setup () {
    error.Clean();
    parameters = std::shared_ptr<r2i::tensorrt::Parameters>( new
                 r2i::tensorrt::Parameters);

    cuda_engine = std::shared_ptr<nvinfer1::ICudaEngine> (new
                  nvinfer1::MockCudaEngine,
                  ICudaEngineDeleter);

    model = std::shared_ptr<r2i::tensorrt::Model>(new r2i::tensorrt::Model);
    engine = std::shared_ptr<r2i::tensorrt::Engine> (new r2i::tensorrt::Engine);
  }

  void teardown () {
  }

};

TEST (TensorRTParameters, ConfigureIncompatibleEngine) {
  std::shared_ptr<r2i::IEngine> mock_engine(new mock::Engine);

  error = parameters->Configure(mock_engine, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (TensorRTParameters, ConfigureIncompatibleModel) {
  std::shared_ptr<r2i::IModel> mock_model(new mock::Model);

  error = parameters->Configure(engine, mock_model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTParameters, ConfigureNullEngine) {
  error = parameters->Configure(nullptr, model);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTParameters, ConfigureNullModel) {
  error = parameters->Configure(engine, nullptr);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTParameters, ConfigureSuccess) {

  error = parameters->Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTParameters, SetAndGetModel) {
  error = parameters->Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IModel> internalModel = parameters->GetModel();

  POINTERS_EQUAL(internalModel.get(), model.get());
}

TEST (TensorRTParameters, SetAndGetEngine) {
  error = parameters->Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  std::shared_ptr<r2i::IEngine> internalEngine = parameters->GetEngine();

  POINTERS_EQUAL(internalEngine.get(), engine.get());
}

TEST (TensorRTParameters, GetVersion) {
  std::string value = "";

  std::shared_ptr<r2i::IEngine> engine(new r2i::tensorrt::Engine);
  std::shared_ptr<r2i::IModel> model(new r2i::tensorrt::Model);

  error = parameters->Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters->Get("version", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  CHECK(! value.compare(value));
}

TEST (TensorRTParameters, GetBatchSize) {
  int value = -1;
  error = parameters->Configure(engine, model);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->SetModel(model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters->Get("batch_size", value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTParameters, SetBatchSize) {
  int in_value = 32;
  int out_value;

  auto tensorrt_model =
    std::dynamic_pointer_cast<r2i::tensorrt::Model, r2i::IModel>(model);

  error = tensorrt_model->SetCudaEngine (cuda_engine);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = engine->SetModel(model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ())

  error = parameters->Configure(engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());;

  error = parameters->Set("batch_size", in_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = parameters->Get("batch_size", out_value);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (in_value, out_value);
}

TEST (TensorRTParameters, GetMissingString) {
  std::string value = "value";

  error = parameters->Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, GetMissingInteger) {
  int value;

  error = parameters->Get("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, SetMissingInteger) {
  int value = 0;

  error = parameters->Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, SetMissingString) {
  std::string value;

  error = parameters->Set("*?\\", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, SetWrongType) {
  int value = 0;

  error = parameters->Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, GetWrongType) {
  int value = 0;

  error = parameters->Set("version", value);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (TensorRTParameters, GetList) {
  std::vector<r2i::ParameterMeta> desc;

  error = parameters->List (desc);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
