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

#ifndef R2I_IPREDICTION_H
#define R2I_IPREDICTION_H

#include <r2i/runtime_error.h>

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   * IPrediction class implements the interface to abstract processed data
   */
  
  class IPrediction {
 public:
    /**
     * \brief Method to Get the prediction for a particular index.
     * \param index Index for a value on the prediction matrix.
     * \param error a RuntimeError with a description of an error.
     * \return a double that indicates the prediction for the provided index .
     */    

    virtual double At (int index,  r2i::RuntimeError &error) {}; 
  };
  
}

#endif // R2I_IPREDICTION_H
