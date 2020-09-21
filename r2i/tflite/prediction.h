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

#ifndef R2I_TFLITE_PREDICTION_H
#define R2I_TFLITE_PREDICTION_H

#include <memory>

#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tflite {

class Prediction: public IPrediction {
 public:
  Prediction ();
  ~Prediction ();
  double At (unsigned int index,  r2i::RuntimeError &error) override;
  void *GetResultData () override;
  unsigned int GetResultSize () override;
  RuntimeError SetTensorValues(float *outputdata, int tensorsize);

 private:
  float *outputdata = NULL;
  int tensorsize;
};

}
}
#endif // R2I_TFLITE_PREDICTION_H
