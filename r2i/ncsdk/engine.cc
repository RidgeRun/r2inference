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

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::ncsdk::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided model is not an NCSDK model");
    return error;
  }

  if (nullptr != this->model) {
    this->model = nullptr;
  }


  this->model = model;

  return error;
}

Engine::Status Engine::GetStatus () {

  return this->status;
}

void Engine::SetStatus (Engine::Status new_status) {

  this->status = new_status;
  return ;
}


RuntimeError Engine::Start ()  {

  ncDeviceHandle_t *device_handle;
  ncGraphHandle_t *model_handle;
  ncFifoHandle_t *input_buffers_ptr;
  ncFifoHandle_t  *output_buffers_ptr;
  ncStatus_t ret;
  unsigned int model_size;
  void *model_ptr;
  Status engine_status;
  unsigned int descriptor_length;
  RuntimeError error;

  engine_status = this->GetStatus();

  if ( Status:: START == engine_status) {
    error.Set (RuntimeError::Code:: WRONG_ENGINE_STATE,
               "Engine in wrong State");
    return error;
  }

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  device_handle = this->movidius_device.get();

  ret = ncDeviceCreate(0, &device_handle);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               GetStringFromStatus (ret, error));
    goto create_fail;
  }

  ret = ncDeviceOpen(device_handle);

  if (NC_OK != ret ) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               GetStringFromStatus (ret, error));
    goto open_fail;
  }

  model_size = this->model->GetDataSize();
  model_ptr = this->model->GetData();
  model_handle = this->model->GetHandler();

  ret = ncGraphAllocate(device_handle, model_handle,
                        model_ptr, model_size);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto graph_fail;
  }

  descriptor_length  = sizeof(struct ncTensorDescriptor_t);

  ret = ncGraphGetOption(model_handle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS,
                         &this->input_descriptor,  &descriptor_length);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto parameters_fail;
  }

  ret = ncGraphGetOption(model_handle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS,
                         &this->output_descriptor,  &descriptor_length);

  if (ret != NC_OK ) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto parameters_fail;
  }
  //Init FIFOs
  input_buffers_ptr = this->input_buffers.get();
  output_buffers_ptr = this->output_buffers.get();

  //Do we want to keep generic Names for the fifos?
  ret = ncFifoCreate("FifoIn0", NC_FIFO_HOST_WO, &input_buffers_ptr);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto input_fifo_fail;
  }
  ret = ncFifoAllocate(input_buffers_ptr, device_handle, &this->input_descriptor,
                       2);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto input_fifo_fail;
  }
  ret = ncFifoCreate("FifoOut0", NC_FIFO_HOST_RO, &output_buffers_ptr);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto output_fifo_fail;
  }
  ret = ncFifoAllocate(output_buffers_ptr, device_handle,
                       &this->output_descriptor, 2);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    goto output_fifo_fail;
  }

  this->SetStatus(Status::START);

  return error;

output_fifo_fail:
  ncFifoDestroy(&output_buffers_ptr);
input_fifo_fail:
  ncFifoDestroy(&input_buffers_ptr);
parameters_fail:
  ncGraphDestroy(&model_handle);
graph_fail:
  ncDeviceClose(device_handle);
open_fail:
create_fail:
  ncDeviceDestroy(&device_handle);
  return error;
}

RuntimeError Engine::Stop () {

  ncDeviceHandle_t *device_handle;
  ncFifoHandle_t *input_buffers_ptr;
  ncFifoHandle_t  *output_buffers_ptr;
  Status engine_status;
  RuntimeError error;
  ncStatus_t ret;


  engine_status = this->GetStatus();

  if ( Status:: IDLE == engine_status) {
    error.Set (RuntimeError::Code:: WRONG_ENGINE_STATE,
               "Engine in wrong State");
    return error;
  }

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  input_buffers_ptr = this->input_buffers.get();
  output_buffers_ptr = this->output_buffers.get();


  ret = ncFifoDestroy(&output_buffers_ptr);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    return error;
  }

  ret = ncFifoDestroy(&input_buffers_ptr);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    return error;
  }


  device_handle = this->movidius_device.get();

  ret = ncDeviceClose(device_handle);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    return error;
  }

  this->SetStatus(Status::IDLE);

  return error;
}

std::unique_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame,
    r2i::RuntimeError &error) {
  return nullptr;

}

}
}

