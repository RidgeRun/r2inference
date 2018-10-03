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

#ifndef R2I_RUNTIMEERROR_H
#define R2I_RUNTIMEERROR_H

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the error handling class for r2i library.
 */
class RuntimeError {

 public:
  /**
   * \brief Cleans the RuntimeError result from any previous operation.
   */
  void Clean();

  /**
  * A string with a description of the error.
  */
  const std::string description;

  /**
  * A code for the error.
  */
  int code;
};

}

#endif // R2I_RUNTIMEERROR_H
