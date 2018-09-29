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

#include <CppUTest/TestHarness.h>
#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/parameters.h>

class MockEngine : public r2i::IEngine {
  void SetModel (const r2i::IModel &in_model,
                 r2i::RuntimeError &error) override {}

  void Start (r2i::RuntimeError &error) override {}

  void Stop (r2i::RuntimeError &error) override {}

  std::unique_ptr<r2i::IPrediction> Predict (const r2i::IFrame &in_frame,
      r2i::RuntimeError &error) override {
    return nullptr;
  }
};

class MockModel : public r2i::IModel {
};

/* Stubs for MVNC */
int stubint = -1;
bool shoulderror = false;
ncStatus_t ncGlobalSetOption(int option, const void *data,
                             unsigned int dataLength) {
  switch (option) {
    case (NC_RW_LOG_LEVEL): {
      stubint = *((int *)data);
      LONGS_EQUAL (sizeof (int), dataLength);
      break;
    }
    default: {
      FAIL ("Unkown flag");
    }
  }

  return shoulderror ? NC_INVALID_PARAMETERS : NC_OK;
}

TEST_GROUP (NcsdkParameters) {
  r2i::RuntimeError error;
  r2i::ncsdk::Parameters params;
  std::shared_ptr<r2i::IEngine> engine;
  std::shared_ptr<r2i::IModel> model;

  void setup () {
    stubint = -1;
    shoulderror = false;
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER, "");
    engine = std::make_shared<MockEngine> ();
    model = std::make_shared<MockModel> ();
  }

  void teardown () {
  }
};

TEST (NcsdkParameters, Configure) {
  params.Configure (engine, model, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);
}

TEST (NcsdkParameters, ConfigureNullEngine) {
  params.Configure (nullptr, model, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.code);
}

TEST (NcsdkParameters, ConfigureNullModel) {
  params.Configure (engine, nullptr, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.code);
}

TEST (NcsdkParameters, ConfigureGetNullEngine) {
  std::shared_ptr<r2i::IEngine> totest;

  totest = params.GetEngine (error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);

  POINTERS_EQUAL (nullptr, totest.get ());
}

TEST (NcsdkParameters, ConfigureGetNullModel) {
  std::shared_ptr<r2i::IModel> totest;

  totest = params.GetModel (error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);

  POINTERS_EQUAL (nullptr, totest.get ());
}

TEST (NcsdkParameters, ConfigureGetEngineModel) {
  std::shared_ptr<r2i::IEngine> enginetotest;
  std::shared_ptr<r2i::IModel> modeltotest;

  params.Configure (engine, model, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);

  enginetotest = params.GetEngine (error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);
  POINTERS_EQUAL (engine.get(), enginetotest.get());

  modeltotest = params.GetModel (error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);
  POINTERS_EQUAL (model.get(), modeltotest.get());
}

TEST (NcsdkParameters, SetGlobalInt) {
  int expected = 2;

  params.Set ("log-level", expected, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.code);

  LONGS_EQUAL (expected, stubint);
}

TEST (NcsdkParameters, SetGlobalIntNotFound) {
  params.Set ("not-found", 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, error.code);

  // Test that stub is not called
  LONGS_EQUAL (-1, stubint);
}

TEST (NcsdkParameters, SetGlobalIntError) {
  shoulderror = true;

  params.Set ("log-level", 0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, error.code);
}
