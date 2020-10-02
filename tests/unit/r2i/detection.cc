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

#include <r2i/detection.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (DetectionOutput) {
  r2i::RuntimeError error;
  r2i::Detection detection;

  void setup() {
    error.Clean();
  }
};

TEST (DetectionOutput, SetAndGetDetections) {
  std::vector< r2i::DetectionInstance > in_value;
  std::vector< r2i::DetectionInstance > out_value;
  r2i::DetectionInstance detection1;
  r2i::Classification class1;
  std::vector< r2i::ClassificationInstance > labels1;

  r2i::BBox box1 = {.x = 0, .y = 0, .width = 100, .height = 100};

  // Set detection labels
  labels1.push_back( std::make_tuple(0, 0.5) );
  labels1.push_back( std::make_tuple(1, 0.5) );
  class1.SetLabels(labels1);

  // Add detection to vector
  detection1.box = box1;
  detection1.labels = class1;
  in_value.push_back(detection1);

  error = detection.SetDetections(in_value);
  CHECK (r2i::RuntimeError::EOK == error.GetCode());

  out_value = detection.GetDetections();
  LONGS_EQUAL (in_value.size(), out_value.size());
  LONGS_EQUAL (in_value[0].box.x, out_value[0].box.x);
  LONGS_EQUAL (in_value[0].box.y, out_value[0].box.y);
  LONGS_EQUAL (in_value[0].box.width, out_value[0].box.width);
  LONGS_EQUAL (in_value[0].box.height, out_value[0].box.height);
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
