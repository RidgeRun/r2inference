/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <cstring>
#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/parameters.h>

#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

class MockEngine : public r2i::IEngine {
  r2i::RuntimeError SetModel  (std::shared_ptr<r2i::IModel>) override {r2i::RuntimeError error; return error;}

  r2i::RuntimeError  Start () override {r2i::RuntimeError error; return error;}

  r2i::RuntimeError Stop () override {r2i::RuntimeError error; return error;}

  std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>,
      r2i::RuntimeError &error) override {
    return nullptr;
  }
};

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};

TEST_GROUP (NcsdkParameters) {
  r2i::ncsdk::Parameters params;
  std::shared_ptr<r2i::IEngine> engine;
  std::shared_ptr<r2i::IModel> model;

  void setup () {
    engine = std::make_shared<r2i::ncsdk::Engine> ();
    model = std::make_shared<r2i::ncsdk::Model> ();
  }

  void teardown () {
  }
};

TEST (NcsdkParameters, Configure) {
  r2i::RuntimeError error;

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (NcsdkParameters, ConfigureNullEngine) {
  r2i::RuntimeError error;

  error = params.Configure (nullptr, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (NcsdkParameters, ConfigureInvalidEngine) {
  r2i::RuntimeError error;
  std::shared_ptr<MockEngine> engine(new MockEngine);

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_ENGINE, error.GetCode ());
}

TEST (NcsdkParameters, ConfigureNullModel) {
  r2i::RuntimeError error;

  error = params.Configure (engine, nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (NcsdkParameters, ConfigureGetNullEngine) {
  std::shared_ptr<r2i::IEngine> totest;

  totest = params.GetEngine ();
  POINTERS_EQUAL (nullptr, totest.get ());
}

TEST (NcsdkParameters, ConfigureGetNullModel) {
  std::shared_ptr<r2i::IModel> totest;

  totest = params.GetModel ();
  POINTERS_EQUAL (nullptr, totest.get ());
}

TEST (NcsdkParameters, ConfigureGetEngineModel) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IEngine> enginetotest;
  std::shared_ptr<r2i::IModel> modeltotest;

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  enginetotest = params.GetEngine ();
  POINTERS_EQUAL (engine.get(), enginetotest.get());

  modeltotest = params.GetModel ();
  POINTERS_EQUAL (model.get(), modeltotest.get());
}

TEST (NcsdkParameters, SetGetGlobalInt) {
  r2i::RuntimeError error;
  int expected = 123;
  int target;

  error = params.Get ("log-level", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  expected = target + 1;

  error = params.Set ("log-level", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("log-level", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, target);
}

TEST (NcsdkParameters, SetGlobalIntNotFound) {
  r2i::RuntimeError error;

  error = params.Set ("not-found", 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (NcsdkParameters, SetGlobalIntError) {
  r2i::RuntimeError error;

  /* api-version is read-only */
  error = params.Set ("api-version", 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode ());
}

TEST (NcsdkParameters, GetDeviceInt) {
  r2i::RuntimeError error;
  int target;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("device-state", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}
