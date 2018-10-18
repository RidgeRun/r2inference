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
#include <r2i/ncsdk/engine.h>
#include <r2i/ncsdk/frame.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};

/* Stubs for MVNC */
int stub_int = -1;
std::string engine_string;
bool engineerror = false;
bool should_error = false;
bool graph_error_alloc = false;
bool graph_error_get = false;
bool graph_error = false;
bool fifo_error = false;
bool device_error = false;


ncStatus_t ncDeviceCreate(int index,
                          struct ncDeviceHandle_t **deviceHandle) {
  static ncDeviceHandle_t handle;


  if (false == should_error) {
    *deviceHandle = &handle;
  } else {
    *deviceHandle = nullptr;
  }

  return should_error ? NC_INVALID_PARAMETERS : NC_OK;
}
ncStatus_t ncFifoDestroy(struct ncFifoHandle_t **fifo) {
  CHECK (*fifo != nullptr);
  *fifo = nullptr;

  return fifo_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncFifoCreate(const char *name, ncFifoType_t type,
                        struct ncFifoHandle_t **fifoHandle) {
  static ncFifoHandle_t handle;
  *fifoHandle = &handle;

  if (false == fifo_error) {
    *fifoHandle = &handle;
  } else {
    *fifoHandle = nullptr;
  }

  return fifo_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle) {
  CHECK (deviceHandle != nullptr);

  return device_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncDeviceDestroy(struct ncDeviceHandle_t **deviceHandle) {
  CHECK (*deviceHandle != nullptr);
  *deviceHandle = nullptr;

  return device_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncDeviceClose(struct ncDeviceHandle_t *deviceHandle) {
  CHECK (deviceHandle != nullptr);

  return device_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncFifoAllocate(struct ncFifoHandle_t *fifo,
                          struct ncDeviceHandle_t *device,
                          struct ncTensorDescriptor_t *tensorDesc, unsigned int numElem) {
  CHECK (fifo != nullptr);
  CHECK (device != nullptr);
  CHECK (tensorDesc != nullptr);

  return fifo_error ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncFifoReadElem(struct ncFifoHandle_t *fifo,
                          void *outputData,
                          unsigned int *outputDataLen,
                          void **userParam) {
  CHECK (fifo != nullptr);
  CHECK (outputData != nullptr);
  CHECK (outputDataLen != nullptr);
  CHECK (userParam != nullptr);

  return fifo_error ? NC_INVALID_DATA_LENGTH : NC_OK;
}

ncStatus_t ncFifoGetOption(struct ncFifoHandle_t *fifo,
                           int option,
                           void *data,
                           unsigned int *dataLen) {
  CHECK (fifo != nullptr);
  CHECK (data != nullptr);
  CHECK (dataLen != nullptr);

  switch (option) {
    case (NC_RO_FIFO_ELEMENT_DATA_SIZE): {
      *((int *)data) = 16;
      break;
    }
    case (-1): {
      memcpy (data, engine_string.data(), *dataLen);
      break;
    }
    default: {
      FAIL ("Unkown flag");
    }
  }
  return fifo_error ? NC_INVALID_DATA_LENGTH : NC_OK;
}

ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t *fifo,
                           const void *inputTensor,
                           unsigned int *inputTensorLength,
                           void *userParam) {
  CHECK (fifo != nullptr);

  return fifo_error ? NC_INVALID_DATA_LENGTH : NC_OK;
}

ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle,
                           struct ncGraphHandle_t *graphHandle,
                           const void *graphBuffer, unsigned int graphBufferLength) {
  CHECK (deviceHandle != nullptr);
  CHECK (graphHandle != nullptr);

  return graph_error_alloc ? NC_INVALID_DATA_LENGTH : NC_OK;
}
ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t *graphHandle,
                                 struct ncFifoHandle_t **fifoIn,
                                 unsigned int inFifoCount,
                                 struct ncFifoHandle_t **fifoOut,
                                 unsigned int outFifoCount) {
  CHECK (graphHandle != nullptr);
  CHECK (fifoIn != nullptr);
  CHECK (fifoOut != nullptr);

  return graph_error ? NC_INVALID_DATA_LENGTH : NC_OK;
}

ncStatus_t ncGraphGetOption(struct ncGraphHandle_t *graphHandle, int option,
                            void *data,
                            unsigned int *dataLength) {
  CHECK (graphHandle != nullptr);
  CHECK (data != nullptr);
  CHECK (dataLength != nullptr);

  switch (option) {
    case (NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS): {
      *((int *)data) = stub_int;
      break;
    }
    case (NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS): {
      *((int *)data) = stub_int;
      break;
    }
    case (-1): {
      memcpy (data, engine_string.data(), *dataLength);
      break;
    }
    default: {
      FAIL ("Unkown flag");
    }
  }

  return graph_error_get ? NC_INVALID_PARAMETERS : NC_OK;
}

ncStatus_t ncGraphDestroy(struct ncGraphHandle_t **graphHandle) {
  CHECK (graphHandle != nullptr);
  *graphHandle = nullptr;

  return graph_error_alloc ? NC_INVALID_PARAMETERS : NC_OK;
}


TEST_GROUP (NcsdkEngine) {
  r2i::ncsdk::Engine engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;

  void setup () {
    engineerror = false;
    graph_error_alloc = false;
    graph_error_get = false;
    graph_error = false;
    fifo_error = false;
    device_error = false;
    model = std::make_shared<r2i::ncsdk::Model> ();
    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::ncsdk::Frame> ();
  }

  void teardown () {
  }
};

TEST (NcsdkEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (NcsdkEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (NcsdkEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (NcsdkEngine, StartEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (NcsdkEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}


TEST (NcsdkEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (NcsdkEngine, StartEngineGraphAllocateError) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  graph_error_alloc = true;
  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode ());

}

TEST (NcsdkEngine, StartEngineGraphGetOptionError) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  graph_error_get = true;
  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());

}

TEST (NcsdkEngine, StartEngineFifoError) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  fifo_error = true;
  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());

}

TEST (NcsdkEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}


TEST (NcsdkEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}


TEST (NcsdkEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (NcsdkEngine, PredictEngine) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
