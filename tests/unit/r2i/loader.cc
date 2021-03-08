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

#include <r2i/loader.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (Loader) {
  r2i::RuntimeError error;
  r2i::Loader loader;

  void setup() {
    error.Clean();
  }
};

TEST (Loader, LoadPreprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader.LoadPreprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::WRONG_API_USAGE == error.GetCode());
}

TEST (Loader, LoadPreprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader.LoadPreprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::WRONG_API_USAGE == error.GetCode());
}

TEST (Loader, LoadPostprocessingWithNullPath) {
  std::string string_path;

  std::shared_ptr<r2i::IPostprocessing> preprocessing = loader.LoadPostprocessing(
        string_path, error);

  CHECK (r2i::RuntimeError::WRONG_API_USAGE == error.GetCode());
}

TEST (Loader, LoadPostprocessingSetWrongFile) {
  std::shared_ptr<r2i::IPostprocessing> preprocessing = loader.LoadPostprocessing(
        __FILE__, error);

  CHECK (r2i::RuntimeError::WRONG_API_USAGE == error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
