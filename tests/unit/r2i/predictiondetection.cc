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

#include <r2i/predictiondetection.h>

#include <cstring>
#include <fstream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#define INPUTS 3
#define X 0
#define Y 0
#define WIDTH 200
#define HEIGHT 200

TEST_GROUP (PredictionDetection) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::PredictionDetection> prediction =
    std::make_shared<r2i::PredictionDetection>();

  r2i::BBox bbox_1 = {X, Y, WIDTH, HEIGHT};
  r2i::BBox bbox_2 = {X, Y, WIDTH, HEIGHT};
  r2i::BBox bbox_3 = {X, Y, WIDTH, HEIGHT};

  r2i::BBox bbox_data[INPUTS] = {bbox_1, bbox_2, bbox_3};
  r2i::BBox *bboxes = nullptr;

  void setup () {
    bboxes = nullptr;
  }

  void teardown () {
  }
};

TEST (PredictionDetection, SetBBoxesSuccess) {
  r2i::RuntimeError error;

  bboxes = &bbox_data[0];

  error = prediction->SetBoundingBoxes(bboxes, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (PredictionDetection, SetNullBBoxes) {
  r2i::RuntimeError error;

  error = prediction->SetBoundingBoxes(bboxes, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (PredictionDetection, SetZeroSizeBBoxes) {
  r2i::RuntimeError error;

  bboxes = &bbox_data[0];

  error = prediction->SetBoundingBoxes(bboxes, 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_PARAMETERS, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
