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

#include <r2i/postprocessing/top_sort_postprocessing.h>

namespace r2i {

/* Sort tuples in descending order based on the first element of the tuple */
static bool SortDesc(const std::tuple<int, double> &a,
                     const std::tuple<int, double> &b) {
  return (std::get<1>(a) > std::get<1>(b));
}

RuntimeError TopSortPostprocessing::Apply(
  std::vector< std::shared_ptr<r2i::IPrediction> > &predictions,
  std::vector< std::shared_ptr<InferenceOutput> > &outputs) {
  float *prediction_data;
  unsigned int prediction_data_size;
  unsigned int num_predictions;
  RuntimeError error;
  std::shared_ptr<r2i::Classification> classification;
  std::vector< r2i::ClassificationInstance > labels;

  /* Number of predictions */
  num_predictions = predictions.size();

  /* Do top sort on all predictions */
  for (unsigned int i = 0; i < num_predictions; ++i) {
    /* Array of prediction values */
    prediction_data = static_cast<float *>(predictions.at(i)->GetResultData());
    if (!prediction_data) {
      error.Set (RuntimeError::Code::NULL_PARAMETER,
                 "NULL Prediction data");
      return error;
    }

    /* Number of elements in the array */
    prediction_data_size = predictions.at(i)->GetResultSize() / sizeof(float);

    for (unsigned int j = 0; j < prediction_data_size; ++j) {
      /* Fill vector with (index, value) pairs */
      labels.push_back(std::make_tuple(j, prediction_data[j]));
    }

    classification = std::make_shared<r2i::Classification>();
    if (!classification) {
      error.Set (RuntimeError::Code::MODULE_ERROR,
                 "Failed to create Classification instance");
      return error;
    }

    error = classification->SetLabels(labels);
    if (error.IsError ()) {
      return error;
    }

    error = SortPrediction(classification);
    if (error.IsError ()) {
      return error;
    }
    outputs.push_back(classification);
    /* Clear contents of labels vector for reuse, size is now 0 again */
    labels.clear();
  }

  return error;
}

RuntimeError TopSortPostprocessing::SortPrediction (
  std::shared_ptr<r2i::Classification> classification) {
  RuntimeError error;
  std::vector< r2i::ClassificationInstance > labels;

  try {
    labels = classification->GetLabels();

    /* Sort indexes in descending order based on the prediction values */
    std::stable_sort(labels.begin(), labels.end(), SortDesc);

    error = classification->SetLabels(labels);
    if (error.IsError ()) {
      return error;
    }

  } catch (const std::exception &e) {
    error.Set (r2i::RuntimeError::Code::MODULE_ERROR,
               e.what());
    return error;
  }

  return error;
}

}  // namespace r2i

r2i::IPostprocessing *
FactoryMakePostprocessing () {
  return new r2i::TopSortPostprocessing ();
}
