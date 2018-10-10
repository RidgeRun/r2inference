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

double Prediction::At (int index,  RuntimeError &error) {

  return 0.1;

}

void Prediction::SetResult (void *data) {
  this->result = data;
}


}
}
