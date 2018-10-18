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
   * Numerical code describing the different errors. See the
   * description field for more contextual information at runtime.
   */
  enum Code {
    /**
    * Everything went okay
    */
    EOK,

    /**
     * API is not being used properly
     */
    WRONG_API_USAGE,

    /**
     * A mandatory parameter was passed in as null
     */
    NULL_PARAMETER,

    /**
     * Trying to access an invalid framework parameter
     */
    INVALID_FRAMEWORK_PARAMETER,

    /**
     * The provided Model is incompatible with the current operation
     */
    INCOMPATIBLE_MODEL,

    /**
     * Framework-specific error triggered
     */
    FRAMEWORK_ERROR,

    /**
     * Problem handling a file
     */
    FILE_ERROR,

    /**
     * Allocation or memory management error
     */
    MEMORY_ERROR,

    /**
     * The provided engine is incompatible with the current operation
     */
    INCOMPATIBLE_ENGINE,

    /**
     * The Engine is in a invalid state
     */
    WRONG_ENGINE_STATE,

    /**
     * An unknown error has ocurred
     */
    UNKNOWN_ERROR,
  };

  /**
   * \brief Creates a new error and initializes it to OK
   */
  RuntimeError ();

  /**
   * \brief Creates and initializes a new error.
   *
   * \param code The code to set in the error
   * \param description A human readable description for the error
   */
  RuntimeError (Code code, const std::string &description);

  /**
   * \brief Cleans the RuntimeError result from any previous operation.
   */
  void Clean ();

  /**
   * \brief Configures an error with a given code and description
   *
   * \param code The code to set in the error
   * \param description A human readable description for the error
   */
  void Set (Code code, const std::string &description);

  /**
   * Returns a human readable description of the error
   */
  const std::string GetDescription ();

  /**
   * Returns the code configured in the error
   */
  Code GetCode ();

 private:
  std::string description;
  Code code;
};

}

#endif // R2I_RUNTIMEERROR_H
