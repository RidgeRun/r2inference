/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <r2i/onnxrt/loader.h>

#include <onnxruntime/core/common/exceptions.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <iostream>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

static bool incompatible_model = false;
static bool env_allocation_fail = false;
static bool session_options_allocation_fail = false;
static bool session_allocation_fail = false;

// To simulate exceptions thrown by onnxruntime API. Exceptions in this
// API are derived from std::exception.
class OnnxrtExcep : public std::exception {
  virtual const char *what() const throw() { return "ONNXRT exception thrown"; }
} onnxrtexcep;

r2i::onnxrt::Model::Model() {
}

r2i::RuntimeError r2i::onnxrt::Model::Set(
  std::shared_ptr<Ort::Session> onnxrt_session) {
  r2i::RuntimeError error;

  if (incompatible_model) {
    error.Set(r2i::RuntimeError::Code::INCOMPATIBLE_MODEL,
              "Incompatible model");
  }

  return error;
}

// FIXME: We are mocking onnxrt constructors. We simulate exceptions
// thrown constructing these objects.
void r2i::onnxrt::Loader::CreateSession(
  std::shared_ptr<Ort::Env> env, const std::string &name,
  std::shared_ptr<Ort::SessionOptions> options) {
  if (incompatible_model || session_allocation_fail) {
    throw onnxrtexcep;
  }
}

void r2i::onnxrt::Loader::CreateEnv(OrtLoggingLevel log_level,
                                    const std::string &log_id) {
  if (env_allocation_fail) {
    throw onnxrtexcep;
  }
}

void r2i::onnxrt::Loader::CreateSessionOptions() {
  if (session_options_allocation_fail) {
    throw onnxrtexcep;
  }
}

TEST_GROUP(OnnxrtLoader) {
  r2i::RuntimeError error;
  r2i::onnxrt::Loader loader;

  void setup() {
    error.Clean();
    loader = r2i::onnxrt::Loader();
    incompatible_model = false;
    env_allocation_fail = false;
    session_options_allocation_fail = false;
    session_allocation_fail = false;
  }
};

TEST(OnnxrtLoader, WrongApiUsage) {
  std::string empty_string = "";

  /* Attempt to load this file as a valid model */
  auto model = loader.Load(empty_string, error);

  CHECK_TEXT(error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL(r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST(OnnxrtLoader, UnableToOpenFile) {
  std::string non_existent_file = "resources/squeezenet_typo.onnx";

  /* Attempt to load this file as a valid model */
  auto model = loader.Load(non_existent_file, error);

  CHECK_TEXT(error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL(r2i::RuntimeError::Code::FILE_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, EnvAllocationFail) {
  env_allocation_fail = true;
  std::string model_path = "resources/squeezenet.onnx";
  auto model = loader.Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, SessionOptionsAllocationFail) {
  session_options_allocation_fail = true;
  std::string model_path = "resources/squeezenet.onnx";
  auto model = loader.Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, SessionAllocationFail) {
  session_allocation_fail = true;
  std::string model_path = "resources/squeezenet.onnx";
  auto model = loader.Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, LoadInvalidFile) {
  incompatible_model = true;
  std::string model_path = "resources/invalid.onnx";
  auto model = loader.Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, LoadValidFile) {
  incompatible_model = false;
  std::string model_path = "resources/squeezenet.onnx";
  auto model = loader.Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
}

int main(int ac, char **av) {
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
  return CommandLineTestRunner::RunAllTests(ac, av);
}
