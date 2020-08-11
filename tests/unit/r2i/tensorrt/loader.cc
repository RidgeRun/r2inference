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
#include <r2i/tensorrt/model.h>
#include <r2i/tensorrt/loader.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include "mockcudaengine.cc"
#include "mockruntime.cc"

bool context_set_error = false;
bool cuda_engine_set_error = false;

namespace r2i {
namespace tensorrt {
Model::Model () {}

RuntimeError Model::SetContext (std::shared_ptr<nvinfer1::IExecutionContext>
                                context) {
  r2i::RuntimeError error;
  if (context_set_error)
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Unable to load cached context");

  return error;
}

RuntimeError Model::SetCudaEngine (std::shared_ptr<nvinfer1::ICudaEngine>
                                   cuda_engine) {
  r2i::RuntimeError error;
  if (cuda_engine_set_error)
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Unable to load cached engine");

  return error;
}

}
}

TEST_GROUP (TensorRTLoader) {
  r2i::RuntimeError error;
  r2i::tensorrt::Loader loader;

  void setup () {
    error.Clean();
    loader = r2i::tensorrt::Loader();
    incompatible_model = false;
    bad_cached_engine = false;
    context_set_error = false;
    cuda_engine_set_error = false;
    fail_context = false;
    fail_runtime = false;
  }
};

TEST (TensorRTLoader, LoadEmptyFileName) {
  auto model = loader.Load("", error);

  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorRTLoader, WrongFileName) {
  auto model = loader.Load("*\"?", error);

  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorRTLoader, LoadFileWithNoPermissions) {
  auto model = loader.Load("/root", error);

  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorRTLoader, LoadInvalidFile) {
  incompatible_model = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorRTLoader, LoadFailRuntime) {
  fail_runtime = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (TensorRTLoader, LoadFailContext) {
  fail_context = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (TensorRTLoader, BadCachedEngine) {
  bad_cached_engine = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode());
}

TEST (TensorRTLoader, LoadSetContextError) {
  context_set_error = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTLoader, LoadSetEngineError) {
  cuda_engine_set_error = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTLoader, LoadSuccess) {
  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorRTLoader, LoadPreprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader.LoadPreprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorRTLoader, LoadPreprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader.LoadPreprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorRTLoader, LoadPostprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPostprocessing> preprocessing = loader.LoadPostprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorRTLoader, LoadPostprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPostprocessing> preprocessing = loader.LoadPostprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
