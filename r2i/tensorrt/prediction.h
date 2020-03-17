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

#ifndef R2I_TENSORRT_PREDICTION_H
#define R2I_TENSORRT_PREDICTION_H

#include <memory>

#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorrt {

class Prediction: public IPrediction {
 public:
  Prediction ();
  double At (unsigned int index,  r2i::RuntimeError &error) override;
  void *GetResultData () override;
  unsigned int GetResultSize () override;
  RuntimeError SetResultBuffer (void *results, size_t size);

 private:
  std::shared_ptr<void> result_buffer;
  size_t result_size;
  /* int64_t GetRequiredBufferSize (TF_Output output, int64_t *dims, */
  /*                                int64_t num_dims); */
};

}
}
#endif // R2I_TENSORRT_PREDICTION_H
