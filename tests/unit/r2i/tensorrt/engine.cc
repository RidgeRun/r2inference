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
#include <r2i/tensorrt/engine.h>
#include <r2i/imodel.h>
#include <r2i/tensorrt/frame.h>
#include <r2i/iprediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mockcudaengine.cc"

#define INPUTS 3

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {
    r2i::RuntimeError error; return error;
  }
};

bool cudaMallocError = false;

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaMalloc(void **devPtr, size_t size) {
  if (!cudaMallocError) {
    return cudaSuccess;
  } else {
    return cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI
cudaMemcpy(void *dst, const void *src,
           size_t count, enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

__host__ __cudart_builtin__ cudaError_t CUDARTAPI
cudaFree(void *devPtr) {
  return cudaSuccess;
}

template <class T>
static void tensorRTIFaceDeleter (T *p) {
  if (p) {
    p->destroy ();
  }
}

TEST_GROUP (TensorRTEngine) {
  r2i::tensorrt::Engine engine;
  std::shared_ptr<r2i::tensorrt::Model> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;
  r2i::RuntimeError error;

  std::shared_ptr<nvinfer1::IExecutionContext> context;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;

  float matrix[INPUTS];

  void setup () {
    error.Clean();

    wrong_network_data_type = false;
    wrong_num_bindings = false;
    execute_error = false;
    cudaMallocError = false;
    output_binding = false;

    context = std::shared_ptr<nvinfer1::IExecutionContext> (new
              nvinfer1::MockExecutionContext,
              tensorRTIFaceDeleter<nvinfer1::IExecutionContext>);

    cuda_engine = std::shared_ptr<nvinfer1::ICudaEngine> (new
                  nvinfer1::MockCudaEngine,
                  tensorRTIFaceDeleter<nvinfer1::ICudaEngine>);

    model = std::make_shared<r2i::tensorrt::Model> ();
    model->SetContext (context);
    model->SetCudaEngine (cuda_engine);

    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::tensorrt::Frame> ();
    frame->Configure(matrix, INPUTS, 1, r2i::ImageFormat::RGB);

  }
};

TEST (TensorRTEngine, SetModel) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, SetModelNull) {
  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTEngine, SetModelInvalid) {
  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTEngine, SetModelWrongModelDataType) {
  wrong_network_data_type = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTEngine, SetModelWrongNumBindings) {
  wrong_num_bindings = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTEngine, SetModelNoCudaEngine) {
  std::shared_ptr<r2i::tensorrt::Model> unconfigured_model =
    std::make_shared<r2i::tensorrt::Model> ();

  error = engine.SetModel (unconfigured_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTEngine, StartEngine) {
  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, StopEngine) {
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, SetBatchSizeNoModel) {
  error = engine.SetBatchSize (MAX_BATCH_SIZE);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTEngine, SetBatchSizeLargeBatchSize) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.SetBatchSize (MAX_BATCH_SIZE + 1);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTEngine, SetBatchSizeNullBatchSize) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.SetBatchSize (0);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTEngine, SetBatchSizeNegativeBatchSize) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.SetBatchSize (-1);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTEngine, SetBatchSizeSucess) {
  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.SetBatchSize (MAX_BATCH_SIZE);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, PredictExecutionError) {
  std::shared_ptr<r2i::IPrediction> prediction;
  execute_error = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (TensorRTEngine, PredictNoModel) {
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTEngine, PredictNullFrame) {
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (nullptr, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTEngine, PredictEngine) {
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, PredictEngineOutput) {
  std::shared_ptr<r2i::IPrediction> prediction;
  output_binding = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTEngine, PredictEngineOutputCudaMallocError) {
  std::shared_ptr<r2i::IPrediction> prediction;
  output_binding = true;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  cudaMallocError = true;
  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
