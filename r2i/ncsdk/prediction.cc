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

#include <mvnc.h>
#include <unordered_map>

#include "r2i/ncsdk/prediction.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

Prediction::Prediction () {
  this->result_data = nullptr;
  this->result_size = 0;
}

unsigned int Prediction::GetResultSize () {
  return this->result_size;
}

void *Prediction::GetResultData () {
  return this->result_data.get ();
}

double Prediction::At (unsigned int index,  RuntimeError &error) {

  unsigned int n_results;
  float *float_result;

  error.Clean ();

  n_results =  this->GetResultSize() / sizeof(float);
  if (n_results < index ) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Triying to access an non-existing index");
    return 0;
  }

  float_result = static_cast<float *> (this->GetResultData ());

  if (nullptr == float_result) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Prediction result not set yet");
    return 0;
  }

  return float_result[index];

}

RuntimeError Prediction::SetResult (std::shared_ptr<float> data,
                                    unsigned int size) {
  RuntimeError error;

  if (nullptr == data) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to assign a null value as presdiction result");
    return error;
  }
  this->result_data = data;
  this->result_size = size;

  return error;
}


}
}

