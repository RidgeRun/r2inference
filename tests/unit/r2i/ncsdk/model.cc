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

#include <CppUTest/TestHarness.h>
#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/model.h>

#define SIZE_TEST 100

TEST_GROUP (NcsdkModel) {
  r2i::RuntimeError error;
  r2i::ncsdk::Model model;

  void setup () {
  }

  void teardown () {
  }
};

TEST (NcsdkModel, StartEmptyName) {
  error.Clean();
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
  error = model.Stop();
}

TEST (NcsdkModel, Start) {
  error.Clean();
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  error = model.Stop();
}

TEST (NcsdkModel, GetHandleBeforeStart) {
  ncGraphHandle_t *graph_handle = model.GetHandler ();
  POINTERS_EQUAL (graph_handle, NULL);
}

TEST (NcsdkModel, GetHandleAfterStart) {
  error.Clean();
  error = model.Start ("graph");
  ncGraphHandle_t *graph_handle = model.GetHandler ();
  CHECK (graph_handle != NULL);
  error = model.Stop();
}

TEST (NcsdkModel, SetGetDataSize) {
  model.SetDataSize (SIZE_TEST);
  LONGS_EQUAL (SIZE_TEST, model.GetDataSize());
}

TEST (NcsdkModel, SetGetData) {
  void *setdata = (void *) malloc (SIZE_TEST);
  model.SetData (setdata);
  void *getdata = model.GetData();
  POINTERS_EQUAL (setdata, model.GetData());
  free (getdata);
}

TEST (NcsdkModel, SetGetDataOverride) {
  void *setdata1 = (void *) malloc (SIZE_TEST);
  void *setdata2 = (void *) malloc (SIZE_TEST);
  model.SetData (setdata1);
  model.SetData (setdata2);
  void *getdata = model.GetData();
  POINTERS_EQUAL (setdata2, model.GetData());
  free (getdata);
}

TEST (NcsdkModel, StopWithStart) {
  error.Clean();
  error = model.Start ("graph");
  error.Clean();
  error = model.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (NcsdkModel, StopWithoutStart) {
  error.Clean();
  error = model.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}
