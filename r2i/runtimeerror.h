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
     * Functionality has not been implemented
     */
    NOT_IMPLEMENTED,

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
     * The provided Parameters are incompatible with the current operation
     */
    INCOMPATIBLE_PARAMETERS,

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
     * Error with a dynamic loading of a module
     */
    MODULE_ERROR,

    /**
     * The provided engine is incompatible with the current operation
     */
    INCOMPATIBLE_ENGINE,

    /**
     * The Engine is in a invalid state
     */
    WRONG_ENGINE_STATE,

    /**
     * The requested framework is not supported in the current system
     */
    UNSUPPORTED_FRAMEWORK,

    /**
     * The delegate was not built properly
     */
    DELEGATE_ERROR,

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
  const std::string GetDescription () const;

  /**
   * Returns the code configured in the error
   */
  Code GetCode () const;

  /**
   * \brief Checks if the RuntimeError is in an error state
   * \return true if an error ocurred, false otherwise
   */
  bool IsError () const;

 private:
  std::string description;
  Code code;
};

/**
 * \brief overload output stream operator to be used on printf
 * \param os output stream to be extended with error description
 * \param current error to be serialized
 * \return extended output stream
 */
std::ostream &operator<<(std::ostream &os, RuntimeError const &self);

}

#endif // R2I_RUNTIMEERROR_H
