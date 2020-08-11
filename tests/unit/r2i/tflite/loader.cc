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

#include <r2i/r2i.h>
#include <r2i/tflite/loader.h>
#include <fstream>
#include <iostream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>
#include <sys/stat.h>
#include <sys/types.h>

static bool incompatible_model = false;

r2i::tflite::Model::Model () {
}

r2i::RuntimeError r2i::tflite::Model::Start (const std::string &name) {
  return r2i::RuntimeError();
}

r2i::RuntimeError r2i::tflite::Model::Set (
  std::shared_ptr<::tflite::FlatBufferModel>
  tfltmodel) {
  r2i::RuntimeError error;

  if (incompatible_model) {
    error.Set (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, "Incompatible model");
  }

  return error;
}

TEST_GROUP (TensorflowliteLoader) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::tflite::Loader> loader{new r2i::tflite::Loader};

  void setup() {
    error.Clean();
  }
};

TEST (TensorflowliteLoader, EmptyString) {
  std::string empty_string = "";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorflowliteLoader, InvalidPath) {
  std::string non_existent_file = "*\"?";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(non_existent_file, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorflowliteLoader, LoadNonExistentFile) {
  /* Setup */
  incompatible_model = true;

  std::string path = "resources/squeezene.tflite";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(path.c_str(), error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());

  /* Teardown */
  incompatible_model = false;
}

TEST (TensorflowliteLoader, LoadInvalidModel) {
  /* Setup */
  std::string path = "resources/invalid.tflite";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(path.c_str(), error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorflowliteLoader, LoadSuccess) {
  /* Setup */
  std::string path = "resources/squeezenet.tflite";

  /* FIXME this isn't actually sending a correct model */
  auto model = loader->Load(path, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorflowliteLoader, LoadPreprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader->LoadPreprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorflowliteLoader, LoadPreprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader->LoadPreprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorflowliteLoader, LoadPostprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPostprocessing> preprocessing =
    loader->LoadPostprocessing(string_path, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

TEST (TensorflowliteLoader, LoadPostprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPostprocessing> preprocessing =
    loader->LoadPostprocessing(__FILE__, error);

  CHECK (r2i::RuntimeError::EOK != error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
