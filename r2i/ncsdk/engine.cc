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

#include "r2i/ncsdk/engine.h"
#include "r2i/ncsdk/prediction.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

void Engine::SetModel (std::shared_ptr<r2i::IModel> in_model,
                       RuntimeError &error) {
    // Init Model handle
    ncStatus_t ret_code = ncGraphCreate("NCSK", &this->model_handle);

    error.Clean ();

    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   "device does not exist");
        return;
    }

    this->model = in_model;
}

void Engine::Start (RuntimeError &error) {

    ncStatus_t ret_code = ncDeviceCreate(0, &this->movidius_device);
    unsigned int model_size;
    void *model_ptr;
    unsigned int descriptor_length;

    error.Clean ();


    if(ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code::DEVICE_NOT_FOUND,
                   "device does not exist");
        return;
    }

    ret_code = ncDeviceOpen(this->movidius_device);

    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code::DEVICE_NOT_OPENED,
                   "device  can not be opened");
        return;
    }

    model_ptr = this->model->GetFd();
    model_size = this->model->GetSize();

    // Send graph to device
    ret_code = ncGraphAllocate(this->movidius_device, this->model_handle,
                               model_ptr, model_size);


    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code::UNKNOWN_ERROR,
                   "device  can not be opened");
        return;
    }

    descriptor_length  = sizeof(struct ncTensorDescriptor_t);


    ncGraphGetOption(this->model_handle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS,
                     &this->input_descriptor,  &descriptor_length);
    ncGraphGetOption(this->model_handle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS,
                     &this->output_descriptor,  &descriptor_length);

    /* Do we want to keep generic Names for the fifos?*/
    ret_code = ncFifoCreate("FifoIn0", NC_FIFO_HOST_WO, &this->input_buffers);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   "Input Fifo Initialization failed");
        return;

    }
    ret_code = ncFifoAllocate(this->input_buffers, this->movidius_device, &this->input_descriptor, 2);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Input Fifo allocation failed!");
        return;
    }
    ret_code = ncFifoCreate("FifoOut0", NC_FIFO_HOST_RO, &this->output_buffers);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Error - Output Fifo Initialization failed!");
        return;
    }
    ret_code = ncFifoAllocate(this->output_buffers, this->movidius_device, &this->output_descriptor, 2);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Output Fifo allocation failed!");
        return;
    }
}

void Engine::Stop (RuntimeError &error) {
    ncStatus_t ret_code = ncDeviceClose(this->movidius_device);

    error.Clean ();

    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Close device failed!");
        return;

    }

    this->movidius_device = NULL;
}

std::unique_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame> in_frame,
        r2i::RuntimeError &error) {

    ncStatus_t ret_code;
    unsigned int input_data_size;
    unsigned int output_data_size;
    unsigned int output_data_size_byte_length;
    void *userParam;
    void *result;

    error.Clean ();

    data = in_frame->GetData();
    input_data_size = in_frame->GetSize();

    //Send the Buffer to the FIFO
    ret_code = ncFifoWriteElem(this->input_buffers, data, &input_data_size, 0);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Failed to write element to Fifo");
        return NULL;
    }

    //Queue inference
    ret_code = ncGraphQueueInference(this->model_handle,
                                     &this->input_buffers, 1,
                                     &this->output_buffers, 1);
    //Read Results
    output_data_size_byte_length = sizeof(unsigned int);

    ret_code = ncFifoGetOption(this->output_buffers, NC_RO_FIFO_ELEMENT_DATA_SIZE, &output_data_size,
                               &output_data_size_byte_length);
    if (ret_code || output_data_size_byte_length != sizeof(unsigned int)) {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Failed to get parameters");
        return NULL;
    }

    /*Don't really want to malloc this...
      do we want special memory at some point?
     */

    result = malloc(output_data_size);
    if (!result) {
        printf("malloc failed!\n");
        return NULL;
    }

    ret_code = ncFifoReadElem(this->output_buffers,
                              result, &output_data_size,
                              &userParam);
    if (ret_code != NC_OK)
    {
        error.Set (RuntimeError::Code:: UNKNOWN_ERROR,
                   " Read Inference result failed!");

        return NULL;

    }

    std::unique_ptr<ncsdk::Prediction> prediction =  std::unique_ptr<ncsdk::Prediction>();

    prediction->SetData(result,error);

    return  prediction;
}

}
}

