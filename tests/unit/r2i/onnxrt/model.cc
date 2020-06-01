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

#include <memory>
#include <r2i/r2i.h>
#include <r2i/onnxrt/model.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (OnnxrtModel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Model model;

  void setup () {
    error.Clean();
    model = r2i::onnxrt::Model();
  }
};

TEST (OnnxrtModel, StartEmptyName) {
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
