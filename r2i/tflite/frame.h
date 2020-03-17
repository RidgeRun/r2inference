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

#ifndef R2I_TFLITE_FRAME_H
#define R2I_TFLITE_FRAME_H

#include <r2i/iframe.h>

namespace r2i {
namespace tflite {

class Frame : public IFrame {
 public:
  Frame ();

  RuntimeError Configure (void *in_data, int width,
                          int height, r2i::ImageFormat::Id format,
                          r2i::DataType::Id data_type = r2i::DataType::Id::UNKNOWN_DATATYPE) override;

  void *GetData () override;

  int GetWidth () override;

  int GetHeight () override;

  ImageFormat GetFormat () override;

 private:
  float *frame_data;
  int frame_width;
  int frame_height;
  ImageFormat frame_format;

};

}
}

#endif //R2I_TFLITE_FRAME_H
