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

#ifndef R2I_TENSORFLOW_PREDICTION_H
#define R2I_TENSORFLOW_PREDICTION_H

#include <memory>
#include <tensorflow/c/c_api.h>

#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorflow {

class Prediction: public IPrediction {
 public:
  Prediction ();
  ~Prediction ();
  double At (unsigned int output_index, unsigned int index,
             r2i::RuntimeError &error) override;
  RuntimeError AddResults (float *data, unsigned int size) override;
  void *GetResultData (unsigned int output_index, RuntimeError &error) override;
  unsigned int GetResultSize (unsigned int output_index,
                              RuntimeError &error) override;

 private:
  std::vector<std::shared_ptr<float []>> results_data;
  std::vector<unsigned int> results_sizes;
};

}
}
#endif // R2I_TENSORFLOW_PREDICTION_H
