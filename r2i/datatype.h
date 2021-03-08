/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#ifndef R2I_DATATYPE_H
#define R2I_DATATYPE_H

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the Data Type handling class for r2i library.
 */
class DataType {

 public:

  /**
   * Numerical id describing the different data types. See the
   * description field for more contextual information at runtime.
   */
  enum Id {
    /**
     * Integer 32 bits per pixel
     */
    INT32,

    /**
     * Unsigned integer 32 bits per pixel
     */
    UINT32,

    /**
     * Float 32 bits per pixel
     */
    FLOAT,


    /**
     * Float 16 bits per pixel
     */
    HALF,


    /**
     * Boolean 8 bits per pixel
     */
    BOOL,


    /**
     * Integer 8 bits per pixel
     */
    INT8,

    /**
     * Unsigned integer 8 bits per pixel
     */
    UINT8,

    UNKNOWN_DATATYPE,
  };

  /**
   * \brief Creates and initializes a data type to Unknown data type.
   */
  DataType ();

  /**
   * \brief Creates and initializes a data type.
   *
   * \param id The id to set in the data type
   */
  DataType (Id id);

  /**
   * Returns the id of the data type
   */
  Id GetId ();

  /**
   * Returns a human readable description of the data type
   */
  const std::string GetDescription ();

  /**
   * Returns the number of bites per pixel
   */
  int GetBytesPerPixel ();

 private:
  Id id;
};

}

#endif // R2I_DATATYPE_H
