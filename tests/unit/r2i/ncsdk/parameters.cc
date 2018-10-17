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
#include <CppUTest/CommandLineTestRunner.h>

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

/* Stubs for ncsdk */

extern "C" {

  int ncGlobalOptionInt = -1;
  bool ncGlobalOptionError = false;

  ncStatus_t ncGlobalGetOption (int param, void *target,
                                unsigned int *target_size) {
    CHECK (nullptr != target);

    if (NC_RW_LOG_LEVEL == param) {
      LONGS_EQUAL (sizeof (int), *target_size);
      *((int *)target) = ncGlobalOptionInt;
    } else {
      FAIL ("Unknown parameter");
    }

    return NC_OK;
  }

  ncStatus_t ncGlobalSetOption (int param, const void *target,
                                unsigned int target_size) {
    CHECK (nullptr != target);

    if (true == ncGlobalOptionError) {
      return NC_ERROR;
    }

    if (NC_RW_LOG_LEVEL == param) {
      LONGS_EQUAL (sizeof (int), target_size);
      ncGlobalOptionInt = *((int *)target);
    } else {
      FAIL ("Unknown parameter");
    }
    return NC_OK;
  }

  int ncDeviceOptionInt = -1;
  bool ncDeviceOptionError = false;

  ncStatus_t ncDeviceGetOption (ncDeviceHandle_t *handle, int param, void *target,
                                unsigned int *target_size) {
    CHECK (nullptr != target);

    if (NC_RO_DEVICE_STATE == param) {
      LONGS_EQUAL (sizeof (int), *target_size);
      *((int *)target) = ncDeviceOptionInt;
    } else {
      FAIL ("Unknown parameter");
    }

    return NC_OK;
  }

  ncStatus_t ncDeviceSetOption (ncDeviceHandle_t *handle, int param,
                                const void *target, unsigned int target_size) {
    CHECK (nullptr != target);

    if (NC_RO_DEVICE_STATE == param) {
      LONGS_EQUAL (sizeof (int), target_size);
      ncDeviceOptionInt = *((int *)target);
    } else {
      FAIL ("Unknown parameter");
    }

    return NC_OK;
  }

  int ncFifoOptionInt = -1;
  bool ncFifoOptionError = false;

  ncStatus_t ncFifoGetOption (ncFifoHandle_t *handle, int param, void *target,
                              unsigned int *target_size) {
    CHECK (nullptr != target);

    if (NC_RO_FIFO_CAPACITY == param) {
      LONGS_EQUAL (sizeof (int), *target_size);
      *((int *)target) = ncFifoOptionInt;
    } else {
      FAIL ("Unknown parameter");
    }

    return NC_OK;
  }

  ncStatus_t ncFifoSetOption (ncFifoHandle_t *handle, int param,
                              const void *target, unsigned int target_size) {
    CHECK (nullptr != target);

    if (NC_RO_FIFO_CAPACITY == param) {
      LONGS_EQUAL (sizeof (int), target_size);
      ncFifoOptionInt = *((int *)target);
    } else {
      FAIL ("Unknown parameter");
    }

    return NC_OK;
  }

} // extern C

TEST_GROUP (NcsdkParameters) {
  r2i::ncsdk::Parameters params;
  std::shared_ptr<r2i::IEngine> engine;
  std::shared_ptr<r2i::IModel> model;

  void setup () {
    ncGlobalOptionInt = -1;
    ncGlobalOptionError = false;
    ncDeviceOptionInt = -1;
    ncDeviceOptionError = false;
    ncFifoOptionInt = -1;
    ncFifoOptionError = false;

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

  ncGlobalOptionInt = 10;

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

  ncGlobalOptionError = true;

  error = params.Set ("log-level", 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode ());
}

TEST (NcsdkParameters, SetGetDeviceInt) {
  r2i::RuntimeError error;
  int expected = 123;
  int target;
  ncDeviceHandle_t ncdevice;

  auto ncengine = std::dynamic_pointer_cast<r2i::ncsdk::Engine, r2i::IEngine>
                  (engine);
  ncengine->SetDeviceHandler (&ncdevice);

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  ncDeviceOptionInt = 10;

  error = params.Get ("device-state", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  expected = target + 1;

  error = params.Set ("device-state", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("device-state", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, target);
}

TEST (NcsdkParameters, GetDeviceNoEngine) {
  r2i::RuntimeError error;
  int target = -1;

  error = params.Get ("device-state", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());

  LONGS_EQUAL (-1, target);
}

TEST (NcsdkParameters, GetDeviceNoEngineHandler) {
  r2i::RuntimeError error;
  int target = -1;

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("device-state", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());

  LONGS_EQUAL (-1, target);
}

TEST (NcsdkParameters, SetGetInputFifoInt) {
  r2i::RuntimeError error;
  int expected = 123;
  int target;
  ncFifoHandle_t ncfifo;

  auto ncengine = std::dynamic_pointer_cast<r2i::ncsdk::Engine, r2i::IEngine>
                  (engine);
  ncengine->SetInputFifoHandler (&ncfifo);

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  ncFifoOptionInt = 10;

  error = params.Get ("input-fifo-capacity", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  expected = target + 1;

  error = params.Set ("input-fifo-capacity", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("input-fifo-capacity", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, target);
}

TEST (NcsdkParameters, GetInputFifoNoHandler) {
  r2i::RuntimeError error;
  int target = -1;

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("input-fifo-capacity", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());

  LONGS_EQUAL (-1, target);
}

TEST (NcsdkParameters, SetGetOutputFifoInt) {
  r2i::RuntimeError error;
  int expected = 123;
  int target;
  ncFifoHandle_t ncfifo;

  auto ncengine = std::dynamic_pointer_cast<r2i::ncsdk::Engine, r2i::IEngine>
                  (engine);
  ncengine->SetOutputFifoHandler (&ncfifo);

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  ncFifoOptionInt = 10;

  error = params.Get ("output-fifo-capacity", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  expected = target + 1;

  error = params.Set ("output-fifo-capacity", expected);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("output-fifo-capacity", target);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  LONGS_EQUAL (expected, target);
}

TEST (NcsdkParameters, GetOutputFifoNoHandler) {
  r2i::RuntimeError error;
  int target = -1;

  error = params.Configure (engine, model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = params.Get ("output-fifo-capacity", target);
  fprintf (stderr, "Error is %s\n", error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());

  LONGS_EQUAL (-1, target);
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
