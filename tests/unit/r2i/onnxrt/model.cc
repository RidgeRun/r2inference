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

#include <r2i/onnxrt/model.h>

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define SIZE 1

/* Custom deleter */
template< typename T >
struct ArrayDeleter {
  void operator ()( T const *p) {
    delete[] p;
  }
};

TEST_GROUP(OnnxrtModel) {
  r2i::RuntimeError error;
  r2i::onnxrt::Model model;
  std::shared_ptr<char> model_data;

  void setup() {
    error.Clean();
    model = r2i::onnxrt::Model();
  }
};

TEST(OnnxrtModel, SetModelNull) {
  error = model.SetOnnxrtModel(nullptr, 0);
  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST(OnnxrtModel, SetModelAllocated) {
  model_data = std::shared_ptr<char>(new char[SIZE], ArrayDeleter<char>());
  error = model.SetOnnxrtModel(std::static_pointer_cast<void>(model_data), SIZE);
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
  POINTERS_EQUAL((void *)model_data.get(), (void *)model.GetOnnxrtModel().get());
  LONGS_EQUAL(SIZE, model.GetOnnxrtModelSize());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}

