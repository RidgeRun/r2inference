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

#include "r2i/nnapi/engine.h"


namespace r2i {
namespace nnapi {
Engine::Engine () : tflite::Engine() {
  this->number_of_threads = 1;
}

void Engine::ConfigureDelegate(::tflite::Interpreter *interpreter) {
    RuntimeError error;
    ::tflite::StatefulNnApiDelegate::Options options;
    options.allow_fp16 = true;
    options.allow_dynamic_dimensions = true;
    options.disallow_nnapi_cpu = false;
    options.accelerator_name = "vsi-npu";

    auto delegate = ::tflite::evaluation::CreateNNAPIDelegate(options);

    if (!delegate){
        error.Set (RuntimeError::Code::DELEGATE_ERROR,
               "NNAPI delegate was not well created");;
    } else {
      interpreter->ModifyGraphWithDelegate(std::move(delegate));
    }
}

Engine::~Engine() {
  this->Stop();
  this->interpreter.reset();
}

}
}