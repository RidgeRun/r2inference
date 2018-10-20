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

#ifndef R2I_PARAMETER_META_H
#define R2I_PARAMETER_META_H

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 * Implements the placeholder for parameter information.
 */
struct ParameterMeta {

  /**
   * Flags to identify the parameters access mode
   */
  enum Flags {
    /**
     * The parameter may be read
     */
    READ = 1 << 0,

    /**
     * The parameter may be written
     */
    WRITE = 1 << 1,

    /**
     * The parameter may be read and written. Equivalent to the
     * bitwise OR of READ and WRITE.
     */
    READWRITE = READ | WRITE
  };

  /**
   * An enum to describe the data type of the parameter
   */
  enum Type {
    /**
     * System dependent integer
     */
    INTEGER,

    /**
     * A standard string
     */
    STRING
  };

  /**
   * A string to identify the name of the parameter
   */
  const std::string name;

  /**
  * A string with a description for the parameter
  */
  const std::string description;

  /**
   * A bitwise OR combination of access flags for the parameter
   */
  const int flags;

  /**
   * The type of the data this parameter holds
   */
  const Type type;
};

}

#endif // R2I_PARAMETER_META_H
