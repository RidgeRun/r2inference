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

#include "iframeworkfactory.h"

#include <functional>
#include <unordered_map>

#include "ncsdk/frameworkfactory.h"

namespace r2i {

static std::unique_ptr<IFrameworkFactory> MakeNcsdkFactory (
  RuntimeError &error);

typedef std::function<std::unique_ptr<IFrameworkFactory>(RuntimeError &)>
MakeFactory;
const std::unordered_map<int, MakeFactory> frameworks ({
  {IFrameworkFactory::FrameworkCode::NCSDK, MakeNcsdkFactory},
});

static std::unique_ptr<IFrameworkFactory>
MakeNcsdkFactory (RuntimeError &error) {
  return std::unique_ptr<ncsdk::FrameworkFactory> (new ncsdk::FrameworkFactory);
}

std::unique_ptr<IFrameworkFactory>
IFrameworkFactory::MakeFactory (FrameworkCode code, RuntimeError &error) {
  auto match = frameworks.find (code);

  /* No match found */
  if (match == frameworks.end ()) {
    error.Set (r2i::RuntimeError::Code::UNSUPPORTED_FRAMEWORK,
               "The framework is invalid or not supported in this system");
    return nullptr;
  }

  return match->second (error);
}

} // namespace r2i
