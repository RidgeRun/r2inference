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

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

TEST_GROUP(OnnxrtLoader) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::onnxrt::Loader> loader =
    std::make_shared<r2i::onnxrt::Loader>();

  void setup() {
    error.Clean();
  }
};

TEST(OnnxrtLoader, WrongApiUsage) {
  std::string empty_string = "";

  auto model = loader->Load(empty_string, error);

  CHECK_TEXT(error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL(r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST(OnnxrtLoader, UnableToOpenFile) {
  std::string non_existent_file = "resources/squeezenet_typo.onnx";

  auto model = loader->Load(non_existent_file, error);

  CHECK_TEXT(error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL(r2i::RuntimeError::Code::FILE_ERROR, error.GetCode());
}

TEST(OnnxrtLoader, LoadValidFile) {
  std::string model_path = "resources/squeezenet.onnx";
  auto model = loader->Load(model_path, error);

  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (OnnxrtLoader, LoadPreprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader->LoadPreprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (OnnxrtLoader, LoadPreprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader->LoadPreprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (OnnxrtLoader, LoadPostprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPostprocessing> preprocessing =
    loader->LoadPostprocessing(string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (OnnxrtLoader, LoadPostprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPostprocessing> preprocessing =
    loader->LoadPostprocessing(__FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
