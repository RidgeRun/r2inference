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

#ifndef R2I_TENSORFLOW_FRAME_H
#define R2I_TENSORFLOW_FRAME_H

#include <tensorflow/c/c_api.h>

#include <r2i/iframe.h>

namespace r2i {
namespace tensorflow {

class Frame : public IFrame {
 public:
  Frame ();

  RuntimeError Configure (void *in_data, int width,
                          int height, r2i::ImageFormat::Id format) override;

  void *GetData () override;

  int GetWidth () override;

  int GetHeight () override;

  ImageFormat GetFormat () override;

  std::shared_ptr<TF_Tensor> GetTensor (std::shared_ptr<TF_Graph> graph,
                                        TF_Operation *operation, RuntimeError &error);

  virtual DataType GetDataType () override;

 private:
  float *frame_data;
  int frame_width;
  int frame_height;
  ImageFormat frame_format;
  std::shared_ptr<TF_Tensor> tensor;

  RuntimeError GetTensorShape (std::shared_ptr<TF_Graph> graph,
                               TF_Operation *operation,
                               TF_DataType &type, int64_t **dims,
                               int64_t &num_dims, int64_t &size);
  RuntimeError CreateTensor (TF_DataType type, int64_t dims[], int64_t num_dims,
                             int64_t size);
  RuntimeError Validate (int64_t dims[], int64_t num_dims);
  void HandleGenericDimensions (int64_t dims[], int64_t num_dims);
};

}
}

#endif //R2I_TENSORFLOW_FRAME_H
