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
#include <r2i/tensorflow/loader.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

static bool incompatible_model = false;

r2i::tensorflow::Model::Model () {
}

r2i::RuntimeError r2i::tensorflow::Model::Start (const std::string &name) {
  return r2i::RuntimeError();
}

r2i::RuntimeError r2i::tensorflow::Model::Load (std::shared_ptr<TF_Buffer>
    pbuffer) {
  r2i::RuntimeError error;

  if (incompatible_model) {
    error.Set (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, "Incompatible model");
  }

  return error;
}

TEST_GROUP (TensorflowLoader) {
};

TEST (TensorflowLoader, WrongApiUsage) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::tensorflow::Loader> loader (new r2i::tensorflow::Loader);

  std::string empty_string = "";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (TensorflowLoader, UnableToOpenFile) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::tensorflow::Loader> loader (new r2i::tensorflow::Loader);

  std::string non_existent_file = "*\"?";

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(non_existent_file, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorflowLoader, UnableToReadFile) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::tensorflow::Loader> loader (new r2i::tensorflow::Loader);

  /* This will work as long as make check is run as a regular user */
  std::string empty_string = "/root";

  /* Attempt to load an unreadable file */
  auto model = loader->Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (TensorflowLoader, LoadInvalidFile) {
  /* Setup */
  incompatible_model = true;

  /* Test */
  r2i::RuntimeError error;

  std::shared_ptr<r2i::tensorflow::Loader> loader (new r2i::tensorflow::Loader);

  /* Attempt to load this file as a valid model */
  auto model = loader->Load(__FILE__, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());

  /* Teardown */
  incompatible_model = false;
}

TEST (TensorflowLoader, LoadSuccess) {
  /* Test */
  r2i::RuntimeError error;

  std::shared_ptr<r2i::tensorflow::Loader> loader (new r2i::tensorflow::Loader);

  /* FIXME this isn't actually sending a correct model */
  auto model = loader->Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
