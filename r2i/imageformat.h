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

#ifndef R2I_IMAGEFORMAT_H
#define R2I_IMAGEFORMAT_H

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the Image formats handling class for r2i library.
 */
class ImageFormat {

 public:

  /**
   * Numerical id describing the different formats. See the
   * description field for more contextual information at runtime.
   */
  enum Id {
    /**
    * RGB: 3 planes with 8 bits per pixel
    */
    RGB,

    /**
     * BGR: 3 planes with 8 bits per pixel
     */
    BGR,

    /**
     * Grayscale: 1 plane with 8 bits per pixel
     */
    GRAY,

    /**
     * Unknown format
     */
    UNKNOWN_FORMAT,
  };

  /**
   * \brief Creates and initializes a format to Unknown format.
   */
  ImageFormat ();

  /**
   * \brief Creates and initializes a format.
   *
   * \param id The id to set in the format
   */
  ImageFormat (Id id);

  /**
   * Returns the id of the format
   */
  Id GetId ();

  /**
   * Returns a human readable description of the format
   */
  const std::string GetDescription ();

  /**
   * Returns the planes number of the format
   */
  int GetNumPlanes ();

 private:
  Id id;
};

}

#endif // R2I_IMAGEFORMAT_H
