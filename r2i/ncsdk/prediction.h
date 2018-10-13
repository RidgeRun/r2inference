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
#ifndef R2I_NCSDK_PREDICTION_H
#define R2I_NCSDK_PREDICTION_H

#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace ncsdk {

class Prediction: public IPrediction {
 public:
  Prediction();
  double At (unsigned int index,  r2i::RuntimeError &error) override;

  r2i::RuntimeError SetResult (void *data, unsigned int size);
  void *GetResultData ();
  unsigned int GetResultSize ();

 private:
  void *result_data;
  unsigned int result_size;
};

}
}
#endif // R2I_NCSDK_PREDICTION_H
