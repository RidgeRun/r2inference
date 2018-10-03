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

#ifndef R2I_NCSDK_ENGINE_H
#define R2I_NCSDK_ENGINE_H

#include <r2i/iengine.h>

namespace r2i
{
namespace ncsdk
{

class Engine : public IEngine
{
public:

  void SetModel (std::shared_ptr<r2i::IModel> in_model,
    r2i::RuntimeError &error) = 0;

  void Start (r2i::RuntimeError &error) = 0;

  void Stop (r2i::RuntimeError &error) = 0;

  std::unique_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
    in_frame,
    r2i::RuntimeError &error) = 0;

private:
  std::shared_ptr<r2i::IModel> model;
  struct ncDeviceHandle_t * movidius_device;
  struct ncGraphHandle_t * model_handle;
  struct ncFifoHandle_t * input_buffers;
  struct ncFifoHandle_t * output_buffers;
  struct ncTensorDescriptor_t input_descriptor;
  struct ncTensorDescriptor_t output_descriptor;


};

}
}
#endif //R2I_NCSDK_ENGINE_H
