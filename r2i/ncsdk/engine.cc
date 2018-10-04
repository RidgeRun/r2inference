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

#include <mvnc.h>
#include <unordered_map>

#include "r2i/iprediction.h"
#include "r2i/ncsdk/engine.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

void Engine::SetModel (std::shared_ptr<r2i::IModel> in_model,
                       RuntimeError &error) {

    error.Clean ();

    if (nullptr == in_model)
    {
        error.Set (RuntimeError::Code:: NULL_PARAMETER,
                   "Model passed as null");
        return;
    }

    this->model = in_model;
}

void Engine::Start (RuntimeError &error) {}

void Engine::Stop (RuntimeError &error) {}

std::unique_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame> in_frame,
        r2i::RuntimeError &error) {
    return nullptr;

}

}
}

