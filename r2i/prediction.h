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

#ifndef R2I_PREDICTION_H
#define R2I_PREDICTION_H

#include <r2i/iprediction.h>

#include <memory>

namespace r2i {

class Prediction: public IPrediction {
 public:
  ~Prediction ();
  virtual double At (unsigned int output_index, unsigned int index,
                     r2i::RuntimeError &error) override;
  virtual void *GetResultData (unsigned int output_index,
                               RuntimeError &error) override;
  virtual unsigned int GetResultSize (unsigned int output_index,
                                      RuntimeError &error) override;
  virtual RuntimeError AddResult (float *data, unsigned int size) override;
  virtual RuntimeError InsertResult (unsigned int output_index, float *data,
                                     unsigned int size) override;

 protected:
  std::vector<std::shared_ptr<float>> results_data;
  std::vector<unsigned int> results_sizes;
};

}

#endif // R2I_PREDICTION_H
