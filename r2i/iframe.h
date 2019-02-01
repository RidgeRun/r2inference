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

#ifndef R2I_IFRAME_H
#define R2I_IFRAME_H

#include <r2i/runtimeerror.h>
#include <r2i/imageformat.h>

#include <memory>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 * Implements the interface to abstract the input data to evaluate
 * using an IModel.
 */
class IFrame {
 public:
  /**
   * \brief Configures the frame to pass to the framework
   * \param in_data Pointer to the data to use. The actual data pointer type
   * will depend on the backend being used.
   * \param width Image Width.
   * \param height Image Height.
   * \param format Image format Id (Defined by each framework)
   * \return A RuntimeError with a description of the error.
   */
  virtual RuntimeError Configure (void *in_data, int width,
                                  int height, r2i::ImageFormat::Id format) = 0;

  /**
   * \brief Gets the data set to the Frame. The actual data pointer type
   * will depend on the backend being used.
   * \return A Shared Pointer with the data set to the Frame.
   */
  virtual void *GetData () = 0;

  /**
   * \brief Gets the Image width set to the Frame.
   * \return Integer value with the Image width.
   */
  virtual int GetWidth () = 0;

  /**
   * \brief Gets the Image height set to the Frame.
   * \return Integer value with the Image height.
   */
  virtual int GetHeight () = 0;

  /**
   * \brief Gets the Image format set to the Frame.
   * \return Image format.
   */
  virtual ImageFormat GetFormat () = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IFrame () {};
};

}

#endif // R2I_IFRAME_H
