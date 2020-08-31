/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
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

#include <r2i/datatype.h>
#include <r2i/prediction.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorrt {

class Prediction: public r2i::Prediction {
 public:
  Prediction ();
  RuntimeError AddResult (float *data, unsigned int size) override;
  RuntimeError InsertResult (unsigned int output_index, float *data,
                             unsigned int size) override;
};

}
}
#endif // R2I_TENSORRT_PREDICTION_H
