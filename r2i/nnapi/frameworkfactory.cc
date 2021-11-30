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
#include "engine.h"
#include "frameworkfactory.h"

namespace r2i {
namespace nnapi {

std::unique_ptr<r2i::IEngine> FrameworkFactory::MakeEngine(
    RuntimeError &error) {
  error.Clean();

  return std::unique_ptr<IEngine>(new Engine);
}

r2i::FrameworkMeta FrameworkFactory::GetDescription(RuntimeError &error) {
  const FrameworkMeta meta{
      .code = r2i::FrameworkCode::NNAPI,
      .name = "NNAPI",
      .label = "nnapi",
      .description = "TensorFlow Lite with NNAPI delegate from Android"};

  error.Clean();

  return meta;
}

}  // namespace nnapi
}  // namespace r2i
