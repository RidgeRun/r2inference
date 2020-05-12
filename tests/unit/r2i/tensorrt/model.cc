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

#include <memory>
#include <r2i/r2i.h>
#include <r2i/tensorrt/model.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include "mockcudaengine.cc"

#define GOOD_FILENAME __FILE__

static void IExecutionContextDeleter (nvinfer1::IExecutionContext *p) {
  if (p)
    p->destroy ();
}

static void ICudaEngineDeleter (nvinfer1::ICudaEngine *p) {
  if (p)
    p->destroy ();
}

TEST_GROUP (TensorRTModel) {
  r2i::RuntimeError error;
  r2i::tensorrt::Model model;
  std::shared_ptr<nvinfer1::IExecutionContext> context;
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;

  void setup () {
    error.Clean();
    model = r2i::tensorrt::Model();
    context = std::shared_ptr<nvinfer1::IExecutionContext> (new
              nvinfer1::MockExecutionContext,
              IExecutionContextDeleter);

    cuda_engine = std::shared_ptr<nvinfer1::ICudaEngine> (new
                  nvinfer1::MockCudaEngine,
                  ICudaEngineDeleter);
  }

  void teardown () {
  }
};

TEST (TensorRTModel, Start) {
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, SetCudaEngineSuccess) {
  error = model.SetCudaEngine (cuda_engine);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, SetNullCudaEngineBuffer) {
  error = model.SetCudaEngine (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTModel, SetContextSuccess) {
  error = model.SetContext (context);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorRTModel, SetNullContextBuffer) {
  error = model.SetContext (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorRTModel, GetContextSuccess) {
  error = model.SetContext (context);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  std::shared_ptr<nvinfer1::IExecutionContext> model_context =
    model.GetTRContext ();

  POINTERS_EQUAL (context.get(), model_context.get());
}

TEST (TensorRTModel, GetCudaEngineSuccess) {
  error = model.SetCudaEngine (cuda_engine);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  std::shared_ptr<nvinfer1::ICudaEngine> model_engine =
    model.GetTRCudaEngine ();

  POINTERS_EQUAL (cuda_engine.get(), model_engine.get());
}

TEST (TensorRTModel, GetNullContextBuffer) {
  std::shared_ptr<nvinfer1::IExecutionContext> model_context =
    model.GetTRContext ();
  LONGS_EQUAL (nullptr, model_context.get());
}

TEST (TensorRTModel, GetNullEngineBuffer) {
  std::shared_ptr<nvinfer1::ICudaEngine> model_engine =
    model.GetTRCudaEngine ();
  LONGS_EQUAL (nullptr, model_engine.get());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
