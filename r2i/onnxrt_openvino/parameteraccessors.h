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

#ifndef R2I_ONNXRT_OPENVINO_PARAMETER_ACCESSORS_H
#define R2I_ONNXRT_OPENVINO_PARAMETER_ACCESSORS_H

#include <r2i/runtimeerror.h>
#include <r2i/iparameters.h>

namespace r2i {
namespace onnxrt_openvino {

class HardwareIdAccessor : public r2i::onnxrt::StringAccessor {
 public:
  RuntimeError Set (IParameters &target);
  RuntimeError Get (IParameters &target);
};

} // namespace onnxrt
} // namespace r2i

#endif //R2I_ONNXRT_OPENVINO_PARAMETER_ACCESSORS_H
