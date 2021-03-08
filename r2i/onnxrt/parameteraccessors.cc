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
namespace onnxrt {

RuntimeError LoggingLevelAccessor::Set (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.SetLogLevel(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError LoggingLevelAccessor::Get (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.GetLogLevel(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError IntraNumThreadsAccessor::Set (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.SetIntraNumThreads(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError IntraNumThreadsAccessor::Get (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.GetIntraNumThreads(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError GraphOptLevelAccessor::Set (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.SetGraphOptLevel(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError GraphOptLevelAccessor::Get (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.GetGraphOptLevel(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError LogIdAccessor::Set (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.SetLogId(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

RuntimeError LogIdAccessor::Get (IParameters &target) {
  RuntimeError error;

  try {
    r2i::onnxrt::Parameters &downcast_parameters =
      dynamic_cast<r2i::onnxrt::Parameters &>(target);
    error = downcast_parameters.GetLogId(this->value);
  } catch (const std::bad_cast &e) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
               "The provided engine is not an ONNXRT parameters");
    return error;
  }

  return error;
}

}
}
