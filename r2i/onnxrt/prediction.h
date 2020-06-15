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

#ifndef R2I_ONNXRT_PREDICTION_H
#define R2I_ONNXRT_PREDICTION_H

#include <memory>

#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace onnxrt {

class Prediction: public IPrediction {
 public:
  Prediction ();
  ~Prediction ();
  double At (unsigned int index,  r2i::RuntimeError &error) override;
  void *GetResultData () override;
  unsigned int GetResultSize () override;
  RuntimeError SetTensorValues(float *output_data, int tensor_size);

 private:
  float *output_data = NULL;
  int tensor_size;
};

}  // namespace onnxrt
}  // namespace r2i

#endif  // R2I_ONNXRT_PREDICTION_H
