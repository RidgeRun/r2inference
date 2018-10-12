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

#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/loader.h>
#include <r2i/ncsdk/model.h>
#include <fstream>

#include <CppUTest/TestHarness.h>

TEST_GROUP (NcsdkLoader) {
  r2i::RuntimeError error;
  r2i::ncsdk::Loader loader;
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

TEST (NcsdkLoader, LoadValidFile) {
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkLoader, DoubleLoad) {
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  model = loader.Load ("test_file", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkLoader, LoadEmptyFile) {
  model = loader.Load ("", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkLoader, LoadNonExistentFile) {
  model = loader.Load ("invalid_graph", error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FILE_ERROR, error.GetCode());
}
