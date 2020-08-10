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

#ifndef R2I_ONNXRT_OPENVINO_ENGINE_H
#define R2I_ONNXRT_OPENVINO_ENGINE_H

#include <r2i/onnxrt/engine.h>

namespace r2i {
namespace onnxrt_openvino {

class Engine : public r2i::onnxrt::Engine {
 public:
  Engine ();
  ~Engine ();
  r2i::RuntimeError SetHardwareId (const std::string &log_id);
  const std::string GetHardwareId ();

 private:
  std::string hardware_option;

 protected:
  void AppendSessionOptionsExecutionProvider(Ort::SessionOptions &session_options,
      r2i::RuntimeError &error)
  override;

};

}
}

#endif //R2I_ONNXRT_OPENVINO_ENGINE_H
