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

#include <cstring>
#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/r2i.h>

#define HIGHEST_CLASSIFICATION_VALUE 1.0

namespace mock {

class Prediction: public r2i::IPrediction {
 public:
  Prediction ():
    output_data(nullptr), tensor_size(0) {
  }

  double At (unsigned int index,  r2i::RuntimeError &error) override  {
    error.Clean ();

    if (nullptr == this->output_data) {
      error.Set (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                 "Prediction was not properly initialized");
      return 0;
    }

    unsigned int n_results =  this->GetResultSize() / sizeof(float);
    if (n_results < index) {
      error.Set (r2i::RuntimeError::Code::MEMORY_ERROR,
                 "Triying to access an non-existing index");
      return 0;
    }
    return this->output_data.get()[index];
  }

  void *GetResultData () override  {
    return static_cast<void *>(this->output_data.get());
  }

  unsigned int GetResultSize () override  {
    return this->tensor_size * sizeof(float);
  }

  r2i::RuntimeError SetPredictValues(float *output_data, int data_size)  {
    r2i::RuntimeError error;

    if (nullptr == output_data) {
      error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
                 "Invalid output tensor values passed to prediction");
      return error;
    }

    if (0 == data_size) {
      error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
                 "Invalid tensor size passed to prediction");
      return error;
    }

    this->tensor_size = data_size;

    this->output_data = std::shared_ptr<float>(new float[this->tensor_size],
                        std::default_delete<float[]>());

    std::memcpy(this->output_data.get(), output_data,
                this->tensor_size * sizeof(float));

    return error;
  }

 private:
  std::shared_ptr<float> output_data;
  int tensor_size;
};

}

TEST_GROUP(TopSort) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    std::make_shared<r2i::TopSortPostprocessing>();
  std::shared_ptr<mock::Prediction> prediction1 =
    std::make_shared<mock::Prediction>();
  std::shared_ptr<mock::Prediction> prediction2 =
    std::make_shared<mock::Prediction>();

  std::vector<float> reference_vector_prediction1 = {HIGHEST_CLASSIFICATION_VALUE, 0.7, 0.6, 0.5};
  std::vector<float> reference_vector_prediction2 = {HIGHEST_CLASSIFICATION_VALUE, 0.92, 0.4, 0.2};
  std::vector<std::vector<float>> reference_predictions_vector;

  std::vector<float> prediction_values1 = {0.5, 0.6, HIGHEST_CLASSIFICATION_VALUE, 0.7};
  std::vector<float> prediction_values2 = {0.4, HIGHEST_CLASSIFICATION_VALUE, 0.2, 0.92};
  std::vector<std::shared_ptr<r2i::IPrediction>> predictions;
  std::vector< std::shared_ptr<r2i::InferenceOutput>> inference_outputs;

  void setup() {
    error.Clean();
    prediction1->SetPredictValues(prediction_values1.data(),
                                  prediction_values1.size());
    prediction2->SetPredictValues(prediction_values2.data(),
                                  prediction_values2.size());
    predictions.push_back(prediction1);
    predictions.push_back(prediction2);
    reference_predictions_vector.push_back(reference_vector_prediction1);
    reference_predictions_vector.push_back(reference_vector_prediction2);
  }
};

TEST(TopSort, ApplySuccess) {
  error = postprocessing->Apply(predictions, inference_outputs);

  int num_predictions = inference_outputs.size();

  /* Run postprocessing on multiple predictions */
  for (int i = 0; i < num_predictions; i++) {
    auto output =
      std::dynamic_pointer_cast<r2i::Classification, r2i::InferenceOutput>
      (inference_outputs.at(i));
    int prediction_size = output->GetLabels().size();
    for (int j = 0; j < prediction_size; j++) {
      std::tuple<int, double> label = output->GetLabels().at(j);
      std::vector<float> labels = reference_predictions_vector.at(i);
      double reference_score = labels.at(j);
      double score = std::get<1>(label);
      LONGS_EQUAL(score, reference_score);
    }
  }

  LONGS_EQUAL(r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST(TopSort, NullPredictionData) {
  std::shared_ptr<mock::Prediction> null_prediction =
    std::make_shared<mock::Prediction>();
  std::vector<std::shared_ptr<r2i::IPrediction>> predictions;
  predictions.push_back(null_prediction);

  error = postprocessing->Apply(predictions, inference_outputs);

  LONGS_EQUAL(r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

int main(int ac, char **av) {
  return CommandLineTestRunner::RunAllTests(ac, av);
}
