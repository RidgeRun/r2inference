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

#include "frameworkfactory.h"
#include "engine.h"

#include <edgetpu.h>
#include <tensorflow/lite/version.h>

namespace r2i {
namespace edgetpu {

std::unique_ptr<r2i::IEngine> FrameworkFactory::MakeEngine (
  RuntimeError &error) {
  error.Clean ();

  return std::unique_ptr<IEngine> (new Engine);
}

r2i::FrameworkMeta FrameworkFactory::GetDescription (
  RuntimeError &error) {
  const FrameworkMeta meta {
    .code = r2i::FrameworkCode::EDGETPU,
    .name = "edgetpu",
    .description = "Google's TensorFlow Lite with EdgeTPU support",
    .version = ::edgetpu::EdgeTpuManager::GetSingleton()->Version()
  };

  error.Clean ();

  return meta;
}

} // namespace tflite
} // namespace r2i
