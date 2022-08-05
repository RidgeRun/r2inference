/* Copyright (C) 2021 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#ifndef R2I_NNAPI_ENGINE_H
#define R2I_NNAPI_ENGINE_H

#include <r2i/tflite/engine.h>

namespace r2i {
namespace nnapi {
  
class Engine : public r2i::tflite::Engine {
 public:
  Engine();
  ~Engine();

 protected:
  RuntimeError ConfigureDelegate(::tflite::Interpreter *interpreter) override;
};

}  // namespace nnapi
}  // namespace r2i
#endif  // R2I_NNAPI_ENGINE_H
