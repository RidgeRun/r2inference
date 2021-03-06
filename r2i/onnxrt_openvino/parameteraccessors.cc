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

#include "parameters.h"
#include "parameteraccessors.h"

namespace r2i {
namespace onnxrt_openvino {
RuntimeError HardwareIdAccessor::Set (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt_openvino::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt_openvino::Parameters &>(target);
    error = downcast_parameters.SetHardwareId(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT OpenVINO parameters");
    return error;
  }

  return error;
}

RuntimeError HardwareIdAccessor::Get (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt_openvino::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt_openvino::Parameters &>(target);
    error = downcast_parameters.GetHardwareId(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT OpenVINO parameters");
    return error;
  }

  return error;
}

}
}
