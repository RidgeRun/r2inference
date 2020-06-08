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

#include "r2i/onnxrt/model.h"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include "/usr/local/include/onnxruntime/core/common/exceptions.h"


namespace r2i {
namespace onnxrt {

Model::Model() {

}

RuntimeError Model::Start(const std::string &name) {
  RuntimeError error;

  return error;
}

}  // namespace onnx
}  // namespace r2i
