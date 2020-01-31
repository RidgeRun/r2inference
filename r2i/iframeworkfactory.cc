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

#include "config.h"
#include "ncsdk/frameworkfactory.h"
#include "tensorflow/frameworkfactory.h"
#include "tflite/frameworkfactory.h"

namespace r2i {

#ifdef HAVE_NCSDK
static std::unique_ptr<IFrameworkFactory>
MakeNcsdkFactory (RuntimeError &error) {
  return std::unique_ptr<ncsdk::FrameworkFactory> (new ncsdk::FrameworkFactory);
}
#endif // HAVE_NCSDK

#ifdef HAVE_TENSORFLOW
static std::unique_ptr<IFrameworkFactory>
MakeTensorflowFactory (RuntimeError &error) {
  return std::unique_ptr<tensorflow::FrameworkFactory> (new
         tensorflow::FrameworkFactory);
}
#endif // HAVE_TENSORFLOW

#ifdef HAVE_TFLITE
static std::unique_ptr<IFrameworkFactory>
MakeTfLiteFactory (RuntimeError &error) {
  return std::unique_ptr<tflite::FrameworkFactory> (new
         tflite::FrameworkFactory);
}
#endif // HAVE_TFLITE

typedef std::function<std::unique_ptr<IFrameworkFactory>(RuntimeError &)>
MakeFactory;
const std::unordered_map<int, MakeFactory> frameworks ({

#ifdef HAVE_NCSDK
  {FrameworkCode::NCSDK, MakeNcsdkFactory},
#endif //HAVE_NCSDK

#ifdef HAVE_TENSORFLOW
  {FrameworkCode::TENSORFLOW, MakeTensorflowFactory},
#endif //HAVE_TENSORFLOW

#ifdef HAVE_TFLITE
  {FrameworkCode::TFLITE, MakeTfLiteFactory},
#endif //HAVE_TFLITE

});

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

std::vector<FrameworkMeta>
IFrameworkFactory::List(RuntimeError &error) {
  std::vector<FrameworkMeta> metas;

  for (auto &fw : frameworks) {
    auto factory = IFrameworkFactory::MakeFactory (static_cast<FrameworkCode>
                   (fw.first), error);
    metas.push_back (factory->GetDescription (error));
  }

  return metas;
}

} // namespace r2i
