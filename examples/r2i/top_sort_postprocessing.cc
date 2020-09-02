/*
 * Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <algorithm>
#include <memory>
#include <vector>

#include <r2i/ipostprocessing.h>

class Example: public r2i::IPostprocessing {
 public:
  std::shared_ptr<r2i::IPrediction> Apply(std::shared_ptr<r2i::IPrediction>
                                          prediction,
                                          r2i::RuntimeError &error) override {
    std::shared_ptr<r2i::IPrediction> sorted_prediction;

    error = SortPrediction(prediction);

    sorted_prediction = prediction;

    return sorted_prediction;
  }

 private:
  r2i::RuntimeError SortPrediction (std::shared_ptr<r2i::IPrediction>
                                    prediction) {
    r2i::RuntimeError error;
    std::vector<double> sorted_prediction;

    float *prediction_data = reinterpret_cast<float *>(prediction->GetResultData());
    /* Number of elements in the array */
    unsigned int prediction_data_size = prediction->GetResultSize() / sizeof(float);
    /* Sort in descending fashion, top prediction at the beginning */
    std::sort(prediction_data, prediction_data + prediction_data_size,
              std::greater<float>());
    return error;
  }
};

r2i::IPostprocessing *
FactoryMakePostprocessing () {
  return new Example ();
}
