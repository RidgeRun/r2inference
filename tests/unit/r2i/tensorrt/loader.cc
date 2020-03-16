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

bool model_set_error = false;

namespace r2i {
namespace tensorrt {
Model::Model () {}

RuntimeError Model::Set (std::shared_ptr<nvinfer1::ICudaEngine> tensorrtmodel) {
  r2i::RuntimeError error;
  if (model_set_error)
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
    model_set_error = false;
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

TEST (TensorRTLoader, BadChechedEngine) {
  bad_cached_engine = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode());
}

TEST (TensorRTLoader, LoadSetError) {
  model_set_error = true;

  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorRTLoader, LoadSuccess) {
  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
