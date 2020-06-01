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

#include <r2i/r2i.h>
#include <r2i/onnxrt/loader.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

static bool incompatible_model = false;

r2i::onnxrt::Model::Model () {
}

r2i::RuntimeError r2i::onnxrt::Model::Start (const std::string &name) {
  r2i::RuntimeError error;

  if (incompatible_model) {
    error.Set (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, "Incompatible model");
  }

  return error;
}

TEST_GROUP (OnnxrtLoader) {
  r2i::RuntimeError error;
  r2i::onnxrt::Loader loader;

  void setup () {
    error.Clean();
    loader = r2i::onnxrt::Loader();
    incompatible_model = false;
  }
};

TEST (OnnxrtLoader, WrongApiUsage) {
  std::string empty_string = "";

  /* Attempt to load this file as a valid model */
  auto model = loader.Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode ());
}

TEST (OnnxrtLoader, UnableToOpenFile) {
  std::string non_existent_file = "*\"?";

  /* Attempt to load this file as a valid model */
  auto model = loader.Load(non_existent_file, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (OnnxrtLoader, UnableToReadFile) {
  /* This will work as long as make check is run as a regular user */
  std::string empty_string = "/root";

  /* Attempt to load an unreadable file */
  auto model = loader.Load(empty_string, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode ());
}

TEST (OnnxrtLoader, LoadInvalidFile) {
  /* Setup */
  incompatible_model = true;

  /* Attempt to load this file as a valid model */
  auto model = loader.Load(__FILE__, error);

  CHECK_TEXT (error.IsError(), error.GetDescription().c_str());
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (OnnxrtLoader, LoadSuccess) {
  /* FIXME this isn't actually sending a correct model */
  auto model = loader.Load(__FILE__, error);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
