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

#include "r2i/postprocessing/top_sort_postprocessing.h"

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

TEST_GROUP(TopSort) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    std::make_shared<r2i::TopSortPostprocessing>();

  void setup() {
    error.Clean();
  }
};

TEST(TopSort, ApplySuccess) {
  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
