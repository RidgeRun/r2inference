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

#include "r2i/coral/engine.h"

namespace r2i {
namespace coral {

Engine::Engine () : tflite::Engine() {
  this->number_of_threads = 1;
  this->coral_context = nullptr;
}

void Engine::SetupResolver(::tflite::ops::builtin::BuiltinOpResolver
                           &resolver) {
  resolver.AddCustom(::coral::kCustomOp, ::coral::RegisterCustomOp());
}

void Engine::SetInterpreterContext(::tflite::Interpreter *interpreter) {
  this->coral_context = ::coral::EdgeTpuManager::GetSingleton()->OpenDevice();

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
                                  this->coral_context.get());
}

Engine::~Engine() {
  this->Stop();
  this->interpreter.reset();
  this->coral_context.reset();
}

} //namespace coral
} //namespace r2i
