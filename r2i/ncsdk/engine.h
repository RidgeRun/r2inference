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
#include <r2i/ncsdk/model.h>

namespace r2i {
namespace ncsdk {

class Engine : public IEngine {
 public:

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

  virtual std::unique_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;

  enum Status {
    IDLE,
    START
  };
 private:
  std::shared_ptr<Model> model;
  std::shared_ptr<struct ncDeviceHandle_t> movidius_device;
  std::shared_ptr<struct ncFifoHandle_t> input_buffers;
  std::shared_ptr<struct ncFifoHandle_t>  output_buffers;
  struct ncTensorDescriptor_t input_descriptor;
  struct ncTensorDescriptor_t output_descriptor;
  Status GetStatus ();
  void SetStatus (Status new_status);
  Status status;


};

}
}
#endif //R2I_NCSDK_ENGINE_H