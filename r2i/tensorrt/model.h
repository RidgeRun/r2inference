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

#ifndef R2I_TENSORRT_MODEL_H
#define R2I_TENSORRT_MODEL_H

#include <iostream>
#include <memory>
#include <NvInfer.h>

#include <r2i/imodel.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorrt {

class Model : public IModel {
 public:
  Model ();
  ~Model ();

  RuntimeError Start (const std::string &name) override;

  std::shared_ptr<nvinfer1::IExecutionContext> GetTRContext ();

  std::shared_ptr<nvinfer1::ICudaEngine> GetTRCudaEngine ();

  RuntimeError SetContext (std::shared_ptr<nvinfer1::IExecutionContext> context);

  RuntimeError SetCudaEngine (std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine);

 private:
  std::shared_ptr < nvinfer1::IExecutionContext > context;

  std::shared_ptr < nvinfer1::ICudaEngine > cuda_engine;
};

}
}

#endif //R2I_TENSORRT_MODEL_H
