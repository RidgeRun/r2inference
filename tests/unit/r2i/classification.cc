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

#include <r2i/classification.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#define CLASS_LABEL_INDEX 0
#define CLASS_SCORE_INDEX 1

TEST_GROUP (ClassificationOutput) {
  r2i::RuntimeError error;
  r2i::Classification classification;

  void setup() {
    error.Clean();
  }
};

TEST (ClassificationOutput, SetAndGetLabels) {
  std::vector< r2i::ClassificationInstance > in_value;
  std::vector< r2i::ClassificationInstance > out_value;

  in_value.push_back( std::make_tuple(0, 0.5) );
  in_value.push_back( std::make_tuple(1, 0.5) );

  error = classification.SetLabels(in_value);
  CHECK (r2i::RuntimeError::EOK == error.GetCode());

  out_value = classification.GetLabels();
  LONGS_EQUAL (in_value.size(), out_value.size());
  LONGS_EQUAL (std::get<CLASS_LABEL_INDEX>(in_value[0]),
               std::get<CLASS_LABEL_INDEX>(out_value[0]));
  LONGS_EQUAL (std::get<CLASS_SCORE_INDEX>(in_value[0]),
               std::get<CLASS_SCORE_INDEX>(out_value[0]));
  LONGS_EQUAL (std::get<CLASS_LABEL_INDEX>(in_value[1]),
               std::get<CLASS_LABEL_INDEX>(out_value[1]));
  LONGS_EQUAL (std::get<CLASS_SCORE_INDEX>(in_value[1]),
               std::get<CLASS_SCORE_INDEX>(out_value[1]));
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
