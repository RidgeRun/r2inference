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

#include <r2i/predictionclassification.h>

#include <cstring>
#include <fstream>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#define INPUTS 3

TEST_GROUP (PredictionClassification) {
  r2i::RuntimeError error;

  std::shared_ptr<r2i::PredictionClassification> prediction =
    std::make_shared<r2i::PredictionClassification>();
  float scores_data[INPUTS] = {0.2, 0.4, 0.6};
  int labels_data[INPUTS] = {1, 2, 3};
  float *scores = nullptr;
  int *labels = nullptr;

  void setup () {
    scores = nullptr;
    labels = nullptr;
  }

  void teardown () {
  }
};

TEST (PredictionClassification, SetScoresSuccess) {
  r2i::RuntimeError error;

  scores = &scores_data[0];

  error = prediction->SetScores(scores, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (PredictionClassification, SetNullScores) {
  r2i::RuntimeError error;

  error = prediction->SetScores(scores, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (PredictionClassification, SetZeroSizeScores) {
  r2i::RuntimeError error;

  scores = &scores_data[0];

  error = prediction->SetScores(scores, 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_PARAMETERS, error.GetCode());
}

TEST (PredictionClassification, SetLabelsSuccess) {
  r2i::RuntimeError error;

  labels = &labels_data[0];

  error = prediction->SetLabels(labels, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (PredictionClassification, SetNullLabels) {
  r2i::RuntimeError error;

  error = prediction->SetLabels(labels, INPUTS);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (PredictionClassification, SetZeroSizeLabels) {
  r2i::RuntimeError error;

  labels = &labels_data[0];

  error = prediction->SetLabels(labels, 0);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_PARAMETERS, error.GetCode());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
