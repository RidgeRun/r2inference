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

#include "r2i/ncsdk/model.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  if (name.empty()) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received empty file name");
    return error;
  }

  if (nullptr == this->graph_handler) {
    ncStatus_t ret = ncGraphCreate (name.c_str(), &this->graph_handler);
    if (NC_OK != ret) {
      error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
                 GetStringFromStatus (ret, error));
    }
  } else {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Model already started");
  }

  return error;
}

RuntimeError Model::Stop () {
  RuntimeError error;

  if (nullptr == this->graph_handler) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Model has not been Started");
    return error;
  }

  ncStatus_t ret = ncGraphDestroy (&this->graph_handler);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
  }

  this->graph_handler = nullptr;

  return error;
}

ncGraphHandle_t *Model::GetHandler () {
  return this->graph_handler;
}

void Model::SetHandler (ncGraphHandle_t *handler) {
  this->graph_handler = handler;
}

std::shared_ptr<void> Model::GetData () {
  return this->graph_data;
}

void Model::SetData (std::shared_ptr<void> graph_data) {
  this->graph_data.swap (graph_data);
}

unsigned int Model::GetDataSize () {
  return this->graph_size;
}

void Model::SetDataSize (unsigned int graph_size) {
  this->graph_size = graph_size;
}

Model::~Model () {
  this->Stop ();
}

} // namespace ncsdk
} // namespace r2i
