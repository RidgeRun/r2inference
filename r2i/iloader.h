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

#ifndef R2I_ILOADER_H
#define R2I_ILOADER_H

#include <r2i/imodel.h>
#include <r2i/runtime_error.h>

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   *  ILoader class implements the interface to validate a IModel implementation
   *  for an IEngine implementation
   */
  class ILoader {
    
 public:
    /**
     * \brief Method to check consistency of a trained model.
     * \param in_path a string of the absolute path to the model for evaluation.
     * \param error a RuntimeError with a description of an error.
     * \return a validated IModel for an IEngine.
     */
    virtual r2i::IModel Load (const std::string &in_path, r2i::RuntimeError &error) = 0;
    
  };
  
}

#endif // R2I_ILOADER_H
