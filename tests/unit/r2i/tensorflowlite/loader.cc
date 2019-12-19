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
#include <r2i/tensorflowlite/loader.h>
#include <fstream>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

static bool incompatible_model = false;
static bool pass_fake_model = false;

r2i::tensorflowlite::Model::Model () {
}

r2i::RuntimeError r2i::tensorflowlite::Model::Start (const std::string &name) {
  return r2i::RuntimeError();
}

r2i::RuntimeError r2i::tensorflowlite::Model::Set (std::shared_ptr<TfLiteModel>
    tfltmodel) {
  r2i::RuntimeError error;

  if (incompatible_model) {
    error.Set (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, "Incompatible model");
  }

  return error;
}

void TfLiteModelDelete(TfLiteModel *model) {return; }

TfLiteModel *TfLiteModelCreateFromFile(const char *model_path) {
  TfLiteModel *fake_model = nullptr;

  if (pass_fake_model) {
    fake_model = (TfLiteModel *)model_path;
  }

  return fake_model;
}

TEST_GROUP (TensorflowliteLoader) {
  r2i::RuntimeError error;

};

TEST (TensorflowliteLoader, WrongApiUsage) {
  std::shared_ptr<r2i::tensorflowlite::Loader> loader (new
      r2i::tensorflowlite::Loader);

  std::string empty_string = "";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorflowliteLoader, UnableToOpenFile) {
  std::shared_ptr<r2i::tensorflowlite::Loader> loader (new
      r2i::tensorflowlite::Loader);

  std::string non_existent_file = "*\"?";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(non_existent_file, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorflowliteLoader, UnableToReadFile) {
  std::shared_ptr<r2i::tensorflowlite::Loader> loader (new
      r2i::tensorflowlite::Loader);

  /* This will work as long as make check is run as a regular user */
  std::string empty_string = "/root";

  /* Attempt to load an unreadable file */
  auto model = loader->Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorflowliteLoader, LoadInvalidFile) {
  /* Setup */
  incompatible_model = true;

  std::shared_ptr<r2i::tensorflowlite::Loader> loader (new
      r2i::tensorflowlite::Loader);

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(__FILE__, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());

  /* Teardown */
  incompatible_model = false;
}

TEST (TensorflowliteLoader, LoadSuccess) {
  /* Setup */
  pass_fake_model = true;

  std::shared_ptr<r2i::tensorflowlite::Loader> loader (new
      r2i::tensorflowlite::Loader);

  /* FIXME this isn't actually sending a correct model */
  auto model = loader->Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
