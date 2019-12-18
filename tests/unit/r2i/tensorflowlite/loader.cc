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
#include <r2i/tensorflowlite/model.h>
#include <fstream>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (TensorflowliteLoader) {
  r2i::RuntimeError error;
  r2i::tensorflowlite::Loader loader;
  std::shared_ptr<r2i::IModel> model;
  std::ofstream test_file;

  void setup () {
    test_file.open ("test_file");
    test_file << "This is a test file";
    test_file.close();
  }

  void teardown () {
    remove ("test_file");
  }
};

TEST (TensorflowliteLoader, LoadValidFile) {
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowliteLoader, DoubleLoad) {
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowliteLoader, LoadEmptyFile) {
  model = loader.Load ("", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (TensorflowliteLoader, LoadNonExistentFile) {
  model = loader.Load ("invalid_graph", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
