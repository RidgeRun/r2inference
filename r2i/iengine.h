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

#ifndef R2I_IENGINE_H
#define R2I_IENGINE_H

#include <r2i/iframe.h>
#include <r2i/imodel.h>
#include <r2i/iprediction.h>
#include <r2i/runtimeerror.h>

#include <memory>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to evaluate IFrame data in a IModel.
 */
class IEngine {

 public:
  /**
   * \brief Sets a trained IModel to an IEngine evaluation
   *  interface
   * \param in_model Trained IModel for a particular framework.
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) = 0;

  /**
   * \brief Initializes the IEngine after an IModel was set.
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Start (r2i::RuntimeError &error) = 0;

  /**
   * \brief Deinitializes an IEngine.
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Stop (r2i::RuntimeError &error) = 0;

  /**
   * \brief Performs a prediction on a IEngine framework, based on a
   *  assigned IModel.
   * \param in_frame Input data to perform the inference.
   * \param error [out] RuntimeError with a description of an error.
   * \return IPrediction inference data obtained from evaluating an IFrame
   *  on the assigned IModel.
   */
  virtual std::unique_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame,
      r2i::RuntimeError &error) = 0;
};

}

#endif // R2I_IENGINE_H
