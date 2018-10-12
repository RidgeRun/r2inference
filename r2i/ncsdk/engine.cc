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

#include "r2i/ncsdk/prediction.h"
#include "r2i/ncsdk/engine.h"
#include "r2i/ncsdk/frame.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

ncDeviceHandle_t *Engine::GetDeviceHandler () {
  return this->movidius_device;
}

void Engine::SetDeviceHandler (ncDeviceHandle_t *handler) {
  this->movidius_device = handler;
}

ncFifoHandle_t *Engine::GetInputFifoHandler () {
  return this->input_buffers;
}

void Engine::SetInputFifoHandler (ncFifoHandle_t *handler) {
  this->input_buffers = handler;
}

ncFifoHandle_t *Engine::GetOutputFifoHandler () {
  return this->output_buffers;
}

void Engine::SetOutputFifoHandler (ncFifoHandle_t *handler) {
  this->output_buffers = handler;
}

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
  std::shared_ptr<void> model_ptr;
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

  model->Start ("NSDK");

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
                        model_ptr.get(), model_size);

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

  //Save used pointers
  this->SetInputFifoHandler(input_buffers_ptr);
  this->SetOutputFifoHandler(output_buffers_ptr);
  this->SetDeviceHandler(device_handle);

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
  ncGraphHandle_t *model_handle;
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

  input_buffers_ptr = this->GetInputFifoHandler();
  output_buffers_ptr = this->GetOutputFifoHandler();


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

  model_handle = this->model->GetHandler();

  ret = ncGraphDestroy(&model_handle);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    return error;
  }

  device_handle = this->GetDeviceHandler();

  ret = ncDeviceClose(device_handle);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               GetStringFromStatus (ret, error));
    return error;
  }

  this->SetStatus(Status::IDLE);

  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame,
    r2i::RuntimeError &error) {

  ncStatus_t ret;
  unsigned int input_data_size;
  unsigned int output_data_size;
  unsigned int output_data_size_byte_length;
  ncFifoHandle_t *input_buffers_ptr;
  ncFifoHandle_t *output_buffers_ptr;
  ncGraphHandle_t *model_handle;
  void *userParam;
  void *result;
  void *data;
  Status engine_status;

  error.Clean ();

  std::shared_ptr<Prediction> prediction (new r2i::ncsdk::Prediction());

  auto frame = std::dynamic_pointer_cast<r2i::ncsdk::Frame, r2i::IFrame>
               (in_frame);

  if (nullptr == frame) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided frame is not an NCSDK frame");
    goto engine_error;
  }

  engine_status = this->GetStatus();

  if ( Status:: IDLE == engine_status) {
    error.Set (RuntimeError::Code:: WRONG_ENGINE_STATE,
               "Engine in wrong State");
    goto engine_error;
  }
  data = frame->GetData();
  input_data_size = in_frame->GetSize();

  input_buffers_ptr = this->GetInputFifoHandler();;
  output_buffers_ptr = this->GetOutputFifoHandler();;


  ret = ncFifoWriteElem(input_buffers_ptr, data, &input_data_size, 0);

  if (NC_OK != ret) {
    error.Set (RuntimeError::Code:: FRAMEWORK_ERROR,
               " Failed to write element to Fifo");
    goto exit;
  }

  //Queue inference
  model_handle = this->model->GetHandler();

  ret = ncGraphQueueInference(model_handle,
                              &input_buffers_ptr, 1,
                              &output_buffers_ptr, 1);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code:: FRAMEWORK_ERROR,
               " Failed to write element to Fifo");
    goto exit;
  }

  //Read Results
  output_data_size_byte_length = sizeof(unsigned int);

  ret = ncFifoGetOption(output_buffers_ptr, NC_RO_FIFO_ELEMENT_DATA_SIZE,
                        &output_data_size,
                        &output_data_size_byte_length);
  if (NC_OK != ret || output_data_size_byte_length != sizeof(unsigned int)) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               " Failed to get parameters");
    goto exit;
  }

  result = malloc(output_data_size);

  if (nullptr == result) {
    error.Set (RuntimeError::Code::UNKNOWN_ERROR,
               "Can't alloc data for inference results");
    goto engine_error;
  }


  ret = ncFifoReadElem(output_buffers_ptr,
                       result, &output_data_size,
                       &userParam);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               " Read Inference result failed!");

    goto read_error;

  }

  prediction->SetResult(result);


  return prediction;

read_error:
  free(result);
exit:
engine_error:
  return nullptr;

}

}
}

