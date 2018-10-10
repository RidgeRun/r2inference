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

#ifndef R2I_NCSDK_ENGINE_H
#define R2I_NCSDK_ENGINE_H

#include <functional>
#include <string>
#include <unordered_map>

#include <r2i/iengine.h>

namespace r2i {
namespace ncsdk {

class Engine : public IEngine {
 public:
  void SetModel (const r2i::IModel &in_model,
                 r2i::RuntimeError &error) {}

  void Start (r2i::RuntimeError &error) {}

  void Stop (r2i::RuntimeError &error) {}

  std::unique_ptr<r2i::IPrediction> Predict (const r2i::IFrame &in_frame,
      r2i::RuntimeError &error) {
    return nullptr;
  }
};

} // namespace ncsdk
} // namespace r2i

#endif //R2I_NCSDK_ENGINE_H
