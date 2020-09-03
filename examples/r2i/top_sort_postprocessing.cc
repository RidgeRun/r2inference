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
#include <iostream>
#include <memory>
#include <vector>

#include <r2i/ipostprocessing.h>

class Example: public r2i::IPostprocessing {
 public:
  std::shared_ptr<r2i::IPrediction> Apply(std::shared_ptr<r2i::IPrediction>
                                          prediction,
                                          r2i::RuntimeError &error) override {
    std::shared_ptr<r2i::IPrediction> out_prediction;

    if (!prediction) {
      error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
                 "Null IPrediction parameter");
      return nullptr;
    }

    out_prediction = SortPrediction(prediction, error);

    return out_prediction;
  }

 private:
  /* Sort tuples in descending order based on the first element of the tuple */
  static bool SortDesc(const std::tuple<double, int> &a,
                       const std::tuple<double, int> &b) {
    return (std::get<0>(a) > std::get<0>(b));
  }

  std::shared_ptr<r2i::IPrediction> SortPrediction (
    std::shared_ptr<r2i::IPrediction>
    prediction, r2i::RuntimeError &error) {
    int max_index;
    double max;

    try {
      /* Array of prediction values */
      float *prediction_data = reinterpret_cast<float *>(prediction->GetResultData());
      /* Number of elements in the array */
      unsigned int prediction_data_size = prediction->GetResultSize() / sizeof(float);
      /* Sort indexes in descending fashion, top prediction pair at the beginning */
      std::vector<std::pair<double, int> > index_value;
      /* Store (value, index) pairs in a vector */
      for (unsigned int i = 0; i < prediction_data_size; ++i) {
        index_value.push_back(std::pair<double, int>(prediction_data[i], i));
      }

      /* Sort indexes in descending order based on the prediction values */
      std::stable_sort(index_value.begin(), index_value.end(), SortDesc);

      std::pair<double, int> top_pair = index_value.at(0);
      max_index = top_pair.second;
      max = top_pair.first;

      std::cout << "Highest probability is label "
                << max_index << " (" << max << ")" << std::endl;

    } catch (const std::exception &e) {
      error.Set (r2i::RuntimeError::Code::MODULE_ERROR,
                 e.what());
      return nullptr;
    }

    return prediction;
  }
};

r2i::IPostprocessing *
FactoryMakePostprocessing () {
  return new Example ();
}
