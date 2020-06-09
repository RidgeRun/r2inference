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

#include "frame.h"
#include "loader.h"

namespace r2i {
namespace onnxrt {

std::unique_ptr<r2i::ILoader> FrameworkFactory::MakeLoader (
  RuntimeError &error) {
  error.Clean ();

  return std::unique_ptr<ILoader> (new Loader);
}

std::unique_ptr<r2i::IEngine> FrameworkFactory::MakeEngine (
  RuntimeError &error) {
  error.Clean ();

  return nullptr;
}

std::unique_ptr<r2i::IParameters> FrameworkFactory::MakeParameters (
  RuntimeError &error) {
  error.Clean ();

  return nullptr;
}

std::unique_ptr<r2i::IFrame> FrameworkFactory::MakeFrame (
  RuntimeError &error) {
  error.Clean ();

  return std::unique_ptr<IFrame> (new Frame);
}

r2i::FrameworkMeta FrameworkFactory::GetDescription (RuntimeError &error) {
  const FrameworkMeta meta {
    .code = r2i::FrameworkCode::ONNXRT,
    .name = "ONNX Runtime",
    .description = "Microsoft ONNX Runtime",
    .version = std::to_string(ORT_API_VERSION)
  };

  error.Clean ();

  return meta;
}

} // namespace onnxrt
} // namespace r2i
