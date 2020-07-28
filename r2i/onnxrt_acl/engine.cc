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

#include "r2i/onnxrt_acl/engine.h"

#include <core/providers/acl/acl_provider_factory.h>

#define USE_ARENA 1

static const OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

namespace r2i {
namespace onnxrt_acl {

Engine::Engine () : onnxrt::Engine() {

}

void Engine::AppendSessionOptionsExecutionProvider(Ort::SessionOptions
    &session_options, r2i::RuntimeError &error) {
  OrtStatus *status = NULL;

  status = OrtSessionOptionsAppendExecutionProvider_ACL(session_options,
           USE_ARENA);
  if (status != NULL) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Failed setting Arm Computer Library (ACL) execution provider");
    g_ort->ReleaseStatus(status);
  }
}

Engine::~Engine() {

}

} //namespace onnxrt_acl
} //namespace r2i
