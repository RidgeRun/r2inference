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

#include <memory>
#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/model.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#define SIZE_TEST 100

TEST_GROUP (NcsdkModel) {
  r2i::RuntimeError error;
  r2i::ncsdk::Model model;

  void setup () {
  }

  void teardown () {
  }
};

TEST (NcsdkModel, Start) {
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  error = model.Stop();
}

TEST (NcsdkModel, StartEmptyName) {
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
  error = model.Stop();
}

TEST (NcsdkModel, DoubleStart) {
  error = model.Start ("graph");
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
  error = model.Stop();
}

TEST (NcsdkModel, GetHandleBeforeStart) {
  ncGraphHandle_t *graph_handle = model.GetHandler ();
  POINTERS_EQUAL (graph_handle, NULL);
}

TEST (NcsdkModel, GetHandleAfterStart) {
  error = model.Start ("graph");
  ncGraphHandle_t *graph_handle = model.GetHandler ();
  CHECK (graph_handle != NULL);
  error = model.Stop();
}

TEST (NcsdkModel, StopWithStart) {
  error = model.Start ("graph");
  error = model.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkModel, DoubleStop) {
  error = model.Start ("graph");
  error = model.Stop ();
  error = model.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkModel, StopWithoutStart) {
  error = model.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_API_USAGE, error.GetCode());
}

TEST (NcsdkModel, StartStopGetHandler) {
  ncGraphHandle_t *graph_handler;
  error = model.Start ("graph");
  error = model.Stop ();
  graph_handler = model.GetHandler();
  POINTERS_EQUAL (nullptr, graph_handler);
}

TEST (NcsdkModel, SetGetDataSize) {
  model.SetDataSize (SIZE_TEST);
  LONGS_EQUAL (SIZE_TEST, model.GetDataSize());
}

TEST (NcsdkModel, SetGetData) {
  std::shared_ptr<void> setdata(malloc(SIZE_TEST), free);
  model.SetData (setdata);
  std::shared_ptr<void> getdata = model.GetData();
  CHECK (setdata == getdata);
}

TEST (NcsdkModel, SetGetDataOverride) {
  std::shared_ptr<void> setdata1(malloc(SIZE_TEST), free);
  std::shared_ptr<void> setdata2(malloc(SIZE_TEST), free);
  model.SetData (setdata1);
  model.SetData (setdata2);
  std::shared_ptr<void> getdata = model.GetData();
  CHECK (setdata2 == model.GetData());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
