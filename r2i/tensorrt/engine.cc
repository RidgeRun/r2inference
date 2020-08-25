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

#include "r2i/tensorrt/engine.h"
#include "r2i/tensorrt/frame.h"
#include "r2i/tensorrt/prediction.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <sstream>
#include <vector>

namespace r2i {
namespace tensorrt {

Engine::Engine () : model(nullptr), batch_size(1) {
}

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::tensorrt::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided model is not an TENSORRT model");
    return error;
  }

  if (nullptr == model->GetTRCudaEngine()) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Model must have a cuda engine before being used");
    return error;
  }

  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine =
    model->GetTRCudaEngine();
  if (cuda_engine->getNbBindings () != 2) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Current implementation only supports single input single output scheme");
    return error;
  }

  nvinfer1::DataType in_type = cuda_engine->getBindingDataType (0);
  nvinfer1::DataType out_type = cuda_engine->getBindingDataType (1);
  if (in_type != nvinfer1::DataType::kFLOAT
      || out_type != nvinfer1::DataType::kFLOAT) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Current implementation only supports float32");
    return error;
  }

  this->model = model;

  return error;
}

RuntimeError Engine::Start ()  {
  return RuntimeError();
}

RuntimeError Engine::Stop () {
  return RuntimeError();
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine;
  ImageFormat in_format;
  int data_size;

  error.Clean ();

  if (!this->model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Engine must have its model set before predicting");
    return nullptr;
  }

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (!frame) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Engine received null frame for prediction");
    return nullptr;
  }

  /* Apply preprocessing, if any */
  error =  Preprocess (*frame);
  if (error.IsError ()) {
    return nullptr;
  }

  data_size = frame->GetDataType().GetBytesPerPixel();

  cuda_engine = this->model->GetTRCudaEngine();
  auto prediction = std::make_shared<Prediction>();

  std::vector < void *> buffers;

  int output_size = 1;
  void *buff;
  std::shared_ptr<void> output_buff;
  for (int i = 0; i < cuda_engine->getNbBindings (); ++i) {
    nvinfer1::Dims dims = cuda_engine->getBindingDimensions (i);

    if (cuda_engine->bindingIsInput(i)) {
      buffers.emplace_back (frame->GetData());
    } else {
      output_size = 1;
      for (int d = 0; d < dims.nbDims; ++d) {
        output_size *= dims.d[d];
      }
      output_size *= this->batch_size;
      output_size *= data_size;

      cudaError_t cuda_error = cudaMalloc (&buff, output_size);
      output_buff = std::shared_ptr<void>(buff, cudaFree);
      if (cudaSuccess != cuda_error) {
        error.Set (RuntimeError::Code::MEMORY_ERROR,
                   "Unable to allocate managed buffer");
        return nullptr;
      }

      buffers.emplace_back (buff);
    }
  }

  bool status = this->model->GetTRContext()->execute (this->batch_size,
                buffers.data ());
  if (!status) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Unable to run prediction on model");
    return nullptr;
  }

  prediction->SetResultBuffer(output_buff,
                              output_size / frame->GetDataType().GetBytesPerPixel(),
                              frame->GetDataType());

  /* Apply postprocessing, if any */
  error =  Postprocess (*prediction);
  if (error.IsError ()) {
    return nullptr;
  }

  return prediction;
}

r2i::RuntimeError Engine::SetBatchSize (const int batch_size) {
  r2i::RuntimeError error;
  if (this->model == nullptr) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Model must have been added before setting a batch size");
    return error;
  }
  if (this->model->GetTRCudaEngine()->getMaxBatchSize() < batch_size) {
    std::ostringstream errorStringStream;
    errorStringStream << "Engine was generated with max batch size of: " <<
                      this->model->GetTRCudaEngine()->getMaxBatchSize() <<
                      ". Batch size can't be larger";

    error.Set (RuntimeError::Code::WRONG_API_USAGE, errorStringStream.str());
    return error;
  }
  if (batch_size <= 0) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Batch size must be larger than 0");
    return error;
  }

  this->batch_size = batch_size;
  return error;
};

const int Engine::GetBatchSize () {
  return this->batch_size;
};

Engine::~Engine () {
  this->Stop();
}

} //namespace tensorrt
} //namepsace r2i
