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

#ifndef R2I_ONNXRT_FRAME_H
#define R2I_ONNXRT_FRAME_H

#include <r2i/iframe.h>

#include <core/session/onnxruntime_cxx_api.h>

namespace r2i {
namespace onnxrt {

class Frame : public IFrame {
 public:
  Frame ();

  RuntimeError Configure (void *in_data, int width,
                          int height, r2i::ImageFormat::Id format) override;

  void *GetData () override;

  int GetWidth () override;

  int GetHeight () override;

  ImageFormat GetFormat () override;

  virtual DataType GetDataType () override;

  std::shared_ptr<Ort::Value> GetInputTensor (std::shared_ptr<Ort::Session>
      session, RuntimeError &error);

 private:
  float *frame_data;
  int frame_width;
  int frame_height;
  ImageFormat frame_format;
  std::shared_ptr<Ort::TypeInfo> type_info_ptr;
  std::shared_ptr<Ort::TensorTypeAndShapeInfo> tensor_info_ptr;
  std::shared_ptr<Ort::MemoryInfo> memory_info_ptr;
  std::shared_ptr<Ort::Value> input_tensor_ptr;

  std::vector<int64_t> GetTensorShape();
  RuntimeError ValidateTensorShape (int64_t *shape);
  void CreateTypeInfo(std::shared_ptr<Ort::Session> session, int input_index);
  void CreateTensorTypeAndShapeInfo();
  void CreateMemoryInfo(OrtAllocatorType allocator_type, OrtMemType mem_type);
  void CreateTensor(size_t element_count, int64_t *shape, size_t shape_len);

};

}  // namespace onnxrt
}  // namespace r2i

#endif //R2I_ONNXRT_FRAME_H
