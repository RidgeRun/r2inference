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
#include <memory>
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
   * \param in_frame input Frame with data to be preprocessed.
   * \param out_frame output Frame with processed data.
   * \return Error with a description message.
   */
  virtual RuntimeError Apply(std::shared_ptr<r2i::IFrame> in_frame,
                             std::shared_ptr<r2i::IFrame> out_frame) = 0;

  /**
   * \brief Gets the available image formats that can be processed.
   * \return Vector with all the available formats.
   */
  virtual std::vector<ImageFormat> GetAvailableFormats() = 0;

  /**
   * \brief Gets the available dimensions for the input images
   * \return Vector with tuples (width, height) of the available dimensions.
   */
  virtual std::vector<std::tuple<int, int>> GetAvailableDataSizes() = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPreprocessing () {};
};


} // namespace r2i

extern "C" {

  /**
   * \brief FactoryMakePreprocessing:
   * Returns a newly allocated algorithm to be used by IPreprocessing.
   * \return Pointer to IPreprocessing object.
   */
  G_MODULE_EXPORT r2i::IPreprocessing *FactoryMakePreprocessing (void);
}

#endif // R2I_IPREPROCESSING_H
