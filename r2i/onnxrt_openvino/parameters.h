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

#ifndef R2I_ONNXRT_OPENVINO_PARAMETERS_H
#define R2I_ONNXRT_OPENVINO_PARAMETERS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <r2i/onnxrt_openvino/engine.h>
#include <r2i/onnxrt/parameters.h>

namespace r2i {
namespace onnxrt_openvino {

class Parameters: public r2i::onnxrt::Parameters {
 public:
  Parameters ();
  RuntimeError Configure (std::shared_ptr<IEngine> in_engine,
                          std::shared_ptr<IModel> in_model) override;
  RuntimeError SetHardwareId (std::string &value);
  RuntimeError GetHardwareId (std::string &value);

 private:
  std::shared_ptr <r2i::onnxrt_openvino::Engine> engine;
  ParameterMeta hardware_id_meta = {
    .name = "hardware-id",
    .description = "OpenVINO hardware device id",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::STRING
  };
};

}  // namespace onnxrt_openvino
}  // namespace r2i

#endif //R2I_ONNXRT_OPENVINO_PARAMETERS_H
