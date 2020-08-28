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

#include <r2i/engine.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

static bool preprocessing_fail = false;
static bool postprocessing_fail = false;

/* Mocks for Pre/Post processing modules */
namespace mock {
class Preprocessing : public r2i::IPreprocessing {
 public:
  Preprocessing () {}
  r2i::RuntimeError Apply(r2i::IFrame &data) {
    r2i::RuntimeError error;
    if (preprocessing_fail) {
      error.Set (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
                 "Functionality not implement");
    }
    return error;
  }
  std::vector<r2i::ImageFormat> GetAvailableFormats() {
    std::vector<r2i::ImageFormat> image_format;
    return image_format;
  }
  std::vector<std::vector<int>> GetAvailableDataSizes() {
    std::vector<std::vector<int>> data_sizes;
    return data_sizes;
  }
};

class Postprocessing : public r2i::IPostprocessing {
 public:
  Postprocessing () {}
  r2i::RuntimeError Apply(r2i::IPrediction &prediction) {
    r2i::RuntimeError error;
    if (postprocessing_fail) {
      error.Set (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
                 "Functionality not implement");
    }
    return error;
  }
};
}

TEST_GROUP (Engine) {
  r2i::RuntimeError error;
  r2i::Engine engine;
  std::shared_ptr<r2i::IPreprocessing> preprocessing;
  std::shared_ptr<r2i::IPostprocessing> postprocessing;
  std::shared_ptr<r2i::IFrame> frame;

  void setup() {
    error.Clean();
  }
};

TEST (Engine, SetPreprocessingWithNull) {
  error = engine.SetPreprocessing(preprocessing);

  CHECK (r2i::RuntimeError::Code::NULL_PARAMETER == error.GetCode());
}

TEST (Engine, SetandGetPreprocessing) {
  preprocessing = std::make_shared<mock::Preprocessing>();

  error = engine.SetPreprocessing(preprocessing);

  CHECK (r2i::RuntimeError::Code::EOK == error.GetCode());

  std::shared_ptr<r2i::IPreprocessing> internal_prepropressing =
    engine.GetPreprocessing();

  POINTERS_EQUAL(internal_prepropressing.get(), preprocessing.get());
}

TEST (Engine, SetPostprocessingWithNull) {
  error = engine.SetPostprocessing(postprocessing);

  CHECK (r2i::RuntimeError::Code::NULL_PARAMETER == error.GetCode());
}

TEST (Engine, SetandGetPostprocessing) {
  postprocessing = std::make_shared<mock::Postprocessing>();

  error = engine.SetPostprocessing(postprocessing);

  CHECK (r2i::RuntimeError::Code::EOK == error.GetCode());

  std::shared_ptr<r2i::IPostprocessing> internal_postpropressing =
    engine.GetPostprocessing();

  POINTERS_EQUAL(internal_postpropressing.get(), postprocessing.get());
}

TEST (Engine, EnginePredictPreprocessFail) {
  preprocessing_fail = true;
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  std::shared_ptr<r2i::IPreprocessing> preprocessing =
    std::make_shared<mock::Preprocessing>();

  error = engine.SetPreprocessing(preprocessing);

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
               error.GetCode ());
}

TEST (Engine, EnginePredictPostprocessFail) {
  postprocessing_fail = true;
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    std::make_shared<mock::Postprocessing>();

  error = engine.SetPostprocessing(postprocessing);

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
               error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
