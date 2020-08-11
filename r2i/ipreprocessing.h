/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#ifndef R2I_IPREPROCESSING_H
#define R2I_IPREPROCESSING_H

#include <r2i/iframe.h>
#include <r2i/imageformat.h>
#include <r2i/runtimeerror.h>

#include <gmodule.h>
#include <vector>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to validate an IPreprocessing implementation
 */
class IPreprocessing {

 public:
  /**
   * \brief Apply a custom preprocessin to the input data.
   * \param data Frame input data.
   * \return Error with a description message.
   */
  virtual RuntimeError apply(IFrame &data) = 0;

  /**
   * \brief Gets the availabel image formats that can be processed.
   * \return Vector with all the available formats.
   */
  virtual std::vector<ImageFormat> getAvailableFormats() = 0;

  /**
   * \brief Getsthe available dimensions for the input images
   * \return Vector with a tuples (width, height) of the available dimensions.
   */
  virtual std::vector<std::vector<int>> getAvailableDataSizes() = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPreprocessing () {};
};


} // namespace r2i

extern "C" {

  /**
    * factory_make_preprocessing:
    *
    * Return a newly allocated algorithm to be used by IPreprocessing
    *
    * Returns: A newly allocated algorithm to be used by IPreprocessing
    */
  G_MODULE_EXPORT r2i::IPreprocessing *factory_make_preprocessing (void);
}

#endif // R2I_IPREPROCESSING_H
