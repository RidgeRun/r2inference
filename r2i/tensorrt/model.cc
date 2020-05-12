/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include "r2i/tensorrt/model.h"

namespace r2i {
namespace tensorrt {

Model::Model () {
}

Model::~Model () {
  /* This order should be kept, otherwise a segfault is triggered */
  this->context = nullptr;
  this->cuda_engine = nullptr;
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::SetContext (std::shared_ptr<nvinfer1::IExecutionContext>
                                context) {
  RuntimeError error;

  if (nullptr == context) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to set execution context with null model pointer");
    return error;
  }

  this->context = context;

  return error;
}

RuntimeError Model::SetCudaEngine (std::shared_ptr<nvinfer1::ICudaEngine>
                                   cuda_engine) {
  RuntimeError error;

  if (nullptr == cuda_engine) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to set cuda engine with null model pointer");
    return error;
  }

  this->cuda_engine = cuda_engine;

  return error;
}

std::shared_ptr<nvinfer1::IExecutionContext> Model::GetTRContext () {
  return this->context;
}

std::shared_ptr<nvinfer1::ICudaEngine> Model::GetTRCudaEngine () {
  return this->cuda_engine;
}

} // namespace tensorrt
} // namespace r2i
