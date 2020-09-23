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

#ifndef R2I_PREDICTION_DECORATOR_H
#define R2I_PREDICTION_DECORATOR_H

#include <r2i/prediction.h>

#include <memory>

namespace r2i {

class PredictionDecorator: public Prediction {
 public:
  PredictionDecorator();
  PredictionDecorator(std::shared_ptr<IPrediction> base);
  ~PredictionDecorator();

  double At (unsigned int output_index, unsigned int index,
             r2i::RuntimeError &error) override;
  void *GetResultData (unsigned int output_index, RuntimeError &error) override;
  unsigned int GetResultSize (unsigned int output_index,
                              RuntimeError &error) override;
  RuntimeError AddResult (float *data, unsigned int size) override;
  RuntimeError InsertResult (unsigned int output_index, float *data,
                             unsigned int size) override;
  unsigned int GetOutputCount() override;

 protected:
  std::shared_ptr<IPrediction> base_prediction = nullptr;
};

}

#endif // R2I_PREDICTION_DECORATOR_H