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

#include <r2i/imodel.h>
#include <r2i/iframe.h>
#include <r2i/iprediction.h>
#include <r2i/runtime_error.h>

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   *  IEngine class implements the interface to evaluate IFrame data
   *  in a IModel.
   */
  class IEngine {
    
 public:
    /**
     * \brief Method set an IModel trained model to an IEngine evaluation
     *  interface
     * \param in_model trained IModel for a particular framework.
     * \param error a RuntimeError with a description of an error.
     * \return void.
     */
    virtual void SetModel (r2i::IModel in_model, r2i::RuntimeError &error) {};

    /**
     * \brief Method to initialize the IEngine after an IModel was set.
     * \param error a RuntimeError with a description of an error.
     * \return void.
     */
    virtual void Start (r2i::RuntimeError &error) {};

    /**
     * \brief Method to deinitialize an IEngine.
     * \param error a RuntimeError with a description of an error.
     * \return void.
     */
    virtual void Stop (r2i::RuntimeError &error) {};
    
    /**
     * \brief Method to perform a prediction on a IEngine framework, based on a
     *  the IModel set.
     * \param in_frame input data to perform the prediction.
     * \param error a RuntimeError with a description of an error.
     * \return IPrediction inference data obtained from evaluating in_frame
     *  on the IModel set. 
     */
    virtual r2i::IPrediction Predict (r2i::IFrame in_frame, r2i::RuntimeError &error) {};
    
  };
  
}

#endif // R2I_IENGINE_H
