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

#include "parameteraccessors.h"
#include "statuscodes.h"

namespace r2i {
namespace ncsdk {

static RuntimeError
ValidateAccessorParameters (Parameters *self, void *target,
                            unsigned int *target_size);

static RuntimeError
ValidateEngineAccessorParameters (Parameters *self,
                                  void *target,
                                  unsigned int *target_size);

static RuntimeError
ValidateEngineDeviceAccessorParameters (Parameters *self,
                                        void *target,
                                        unsigned int *target_size);

static RuntimeError
ValidateInputFifoAccessorParameters (Parameters *self,
                                     void *target,
                                     unsigned int *target_size);

static RuntimeError
ValidateOutputFifoAccessorParameters (Parameters *self,
                                      void *target,
                                      unsigned int *target_size);

static RuntimeError ValidateAccessorParameters (Parameters *self, void *target,
    unsigned int *target_size) {
  RuntimeError error;

  if (nullptr == self) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL instance, something really bad is happening");
    return error;
  }

  if (nullptr == target) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL target passed to accessor");
    return error;
  }

  if (nullptr == target_size) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL size passed to accessor");
    return error;
  }

  return error;
}

RuntimeError SetParameterGlobal (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  ncStatus_t ncret = ncGlobalSetOption (param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError GetParameterGlobal (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  ncStatus_t ncret = ncGlobalGetOption (param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateEngineAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  std::shared_ptr<IEngine> engine = self->GetEngine();
  if (nullptr == engine) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Parameters not been configured with a valid engine");
    return error;
  }

  return error;
}

static RuntimeError ValidateEngineDeviceAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  ncDeviceHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                             (self->GetEngine())->GetDeviceHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK device configured");
    return error;
  }

  return error;
}

RuntimeError SetParameterEngine (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineDeviceAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncDeviceHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                             (self->GetEngine())->GetDeviceHandler ();
  ncStatus_t ncret = ncDeviceSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError GetParameterEngine (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineDeviceAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncDeviceHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                             (self->GetEngine())->GetDeviceHandler ();
  ncStatus_t ncret = ncDeviceGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateInputFifoAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetInputFifoHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK input fifo configured");
    return error;
  }

  return error;
}

RuntimeError SetParameterInputFifo (Parameters *self, int param,
                                    void *target,
                                    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateInputFifoAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetInputFifoHandler ();
  ncStatus_t ncret = ncFifoSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError GetParameterInputFifo (Parameters *self, int param,
                                    void *target,
                                    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateInputFifoAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetInputFifoHandler ();
  ncStatus_t ncret = ncFifoGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateOutputFifoAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetOutputFifoHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK output fifo configured");
    return error;
  }

  return error;
}

RuntimeError SetParameterOutputFifo (Parameters *self, int param,
                                     void *target,
                                     unsigned int *target_size) {
  RuntimeError error;

  error = ValidateOutputFifoAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetOutputFifoHandler ();
  ncStatus_t ncret = ncFifoSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError GetParameterOutputFifo (Parameters *self, int param,
                                     void *target,
                                     unsigned int *target_size) {
  RuntimeError error;

  error = ValidateOutputFifoAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (self->GetEngine())->GetOutputFifoHandler ();
  ncStatus_t ncret = ncFifoGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateGraphAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  std::shared_ptr<IModel> model = self->GetModel();
  if (nullptr == model) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Parameters not been configured with a valid model");
    return error;
  }

  ncGraphHandle_t *handle = std::dynamic_pointer_cast<Model, IModel>
                            (self->GetModel())->GetHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK graph configured");
    return error;
  }

  return error;
}

RuntimeError SetParameterGraph (Parameters *self, int param,
                                void *target,
                                unsigned int *target_size) {
  RuntimeError error;

  error = ValidateGraphAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncGraphHandle_t *handle = std::dynamic_pointer_cast<Model, IModel>
                            (self->GetModel())->GetHandler ();
  ncStatus_t ncret = ncGraphSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError GetParameterGraph (Parameters *self, int param,
                                void *target,
                                unsigned int *target_size) {
  RuntimeError error;

  error = ValidateGraphAccessorParameters (self, target, target_size);
  if (error.IsError ()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncGraphHandle_t *handle = std::dynamic_pointer_cast<Model, IModel>
                            (self->GetModel())->GetHandler ();
  ncStatus_t ncret = ncGraphGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

} // namespace ncsdk
} // namespace r2i
