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
std::string stubstring;

bool shoulderror = false;
ncStatus_t ncGlobalSetOption(int option, const void *data,
                             unsigned int dataLength) {
  switch (option) {
    case (NC_RW_LOG_LEVEL): {
      stubint = *((int *)data);
      LONGS_EQUAL (sizeof (int), dataLength);
      break;
    }
    case (-1): {
      stubstring = static_cast<const char *>(data);
      LONGS_EQUAL (dataLength, stubstring.size() + 1);
      break;
    }
    default: {
      FAIL ("Unkown flag");
    }
  }

  return shoulderror ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncGlobalGetOption(int option, void *data,
                             unsigned int *dataLength) {
  switch (option) {
    case (NC_RO_API_VERSION): {
      *((int *)data) = stubint;
      break;
    }
    case (-1): {
      memcpy (data, stubstring.data(), *dataLength);
      break;
    }
    default: {
      FAIL ("Unkown flag");
    }
  }

  return shoulderror ? NC_INVALID_PARAMETERS : NC_OK;
}

TEST_GROUP (NcsdkParameters) {
  r2i::ncsdk::Parameters params;
  std::shared_ptr<r2i::IEngine> engine;
  std::shared_ptr<r2i::IModel> model;

  void setup () {
    stubint = -1;
    shoulderror = false;
    engine = std::make_shared<MockEngine> ();
    model = std::make_shared<MockModel> ();
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

TEST (NcsdkParameters, SetGlobalInt) {
  r2i::RuntimeError error;
  int expected = 2;

  error = params.Set ("log-level", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, stubint);
}

TEST (NcsdkParameters, SetGlobalIntNotFound) {
  r2i::RuntimeError error;

  error = params.Set ("not-found", 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());

  // Test that stub is not called
  LONGS_EQUAL (-1, stubint);
}

TEST (NcsdkParameters, SetGlobalIntError) {
  r2i::RuntimeError error;
  shoulderror = true;

  error = params.Set ("log-level", 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (NcsdkParameters, SetGlobalString) {
  r2i::RuntimeError error;
  const std::string expected = "expected";

  error = params.Set ("mock-param", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  STRCMP_EQUAL (expected.c_str(), stubstring.c_str());
}

TEST (NcsdkParameters, GetGlobalInt) {
  r2i::RuntimeError error;
  int expected = 1234;
  int target;

  stubint = expected;
  error = params.Get ("api-version", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, stubint);
}

TEST (NcsdkParameters, GetGlobalString) {
  r2i::RuntimeError error;
  const std::string expected = "expected";
  std::string target;

  stubstring = expected;
  error = params.Get ("mock-param", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  STRCMP_EQUAL (expected.c_str(), stubstring.c_str());
}
