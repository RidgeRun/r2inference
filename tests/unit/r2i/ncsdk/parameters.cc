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
#include <r2i/r2i.h>
#include <r2i/ncsdk/parameters.h>

class MockEngine : public r2i::IEngine {
  void SetModel (const r2i::IModel &in_model,
                 r2i::RuntimeError &error) override {}

  void Start (r2i::RuntimeError &error) override {}

  void Stop (r2i::RuntimeError &error) override {}

  std::unique_ptr<r2i::IPrediction> Predict (const r2i::IFrame &in_frame,
      r2i::RuntimeError &error) override {
    return nullptr;
  }
};

class MockModel : public r2i::IModel {
};

TEST_GROUP (NcsdkParameters) {
  void setup() {
  }

  void teardown() {
  }
};

TEST (NcsdkParameters, Configure) {
  r2i::RuntimeError error;
  r2i::ncsdk::Parameters params;
  std::shared_ptr<r2i::IEngine> engine (new MockEngine);
  std::shared_ptr<r2i::IModel> model (new MockModel);

  params.Configure (engine, model, error);
}
