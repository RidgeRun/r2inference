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

#include "r2i/tensorflowlite/loader.h"

#include <fstream>
#include <memory>
#include <tensorflow/lite/c/c_api.h>

#include "r2i/imodel.h"
#include "r2i/tensorflowlite/model.h"

namespace r2i {
namespace tensorflowlite {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {

  auto model = std::make_shared<Model>();

  return model;
}
}
}