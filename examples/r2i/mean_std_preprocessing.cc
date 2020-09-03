/*
 * Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <memory>
#include <vector>

#include <r2i/ipreprocessing.h>
#include <r2i/onnxrt/frame.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* Model specific required dimensions */
#define REQ_WIDTH_224 224
#define REQ_HEIGTH_224 224

/* Constants for preprocessing */
#define MEAN 128.0
#define STD_DEV 128.0

class Example: public r2i::IPreprocessing {
 public:
  r2i::RuntimeError Apply(std::shared_ptr<r2i::IFrame> in_frame,
                          std::shared_ptr<r2i::IFrame> out_frame, int required_width, int required_height,
                          r2i::ImageFormat::Id required_format) override {
    r2i::RuntimeError error;
    int width;
    int height;

    this->required_width = required_width;
    this->required_height = required_height;
    this->required_format = required_format;

    width = in_frame->GetWidth();
    height = in_frame->GetHeight();

    std::shared_ptr<float> processed_data = PreProcessImage((
        const unsigned char *)in_frame->GetData(), width, height, this->required_width,
                                            this->required_height);
    error = out_frame->Configure (processed_data.get(), this->required_width,
                                  this->required_height,
                                  r2i::ImageFormat::Id::RGB);
    if (error.IsError ()) {
      return error;
    }
    return error;
  }

  std::vector<r2i::ImageFormat> GetAvailableFormats() override {
    this->formats.push_back(r2i::ImageFormat(r2i::ImageFormat::Id::RGB));
    return this->formats;
  }

  std::vector<std::tuple<int, int>> GetAvailableDataSizes() override {
    this->dimensions.push_back(std::tuple<int, int>(REQ_WIDTH_224, REQ_HEIGTH_224));
    return this->dimensions;
  }

 private:
  int required_width, required_height;
  r2i::ImageFormat required_format;
  std::shared_ptr<float> adjusted_ptr;
  std::shared_ptr<unsigned char> scaled_ptr;
  std::vector<std::tuple<int, int>> dimensions;
  std::vector<r2i::ImageFormat> formats;

  std::shared_ptr<float> PreProcessImage (const unsigned char *input,
                                          int width, int height, int required_width, int required_height) {

    const int channels = 3;
    const int scaled_size = channels * required_width * required_height;
    this->scaled_ptr = std::shared_ptr<unsigned char>(new unsigned
                       char[scaled_size],
                       std::default_delete<const unsigned char[]>());
    this->adjusted_ptr = std::shared_ptr<float>(new float[scaled_size],
                         std::default_delete<float[]>());

    float *adjusted = adjusted_ptr.get();
    unsigned char *scaled = scaled_ptr.get();

    stbir_resize_uint8(input, width, height, 0, scaled, required_width,
                       required_height, 0, channels);

    for (int i = 0; i < scaled_size; i += channels) {
      /* RGB = (RGB - Mean)*StdDev */
      adjusted[i + 0] = (static_cast<float>(scaled[i + 0]) - MEAN) / STD_DEV;
      adjusted[i + 1] = (static_cast<float>(scaled[i + 1]) - MEAN) / STD_DEV;
      adjusted[i + 2] = (static_cast<float>(scaled[i + 2]) - MEAN) / STD_DEV;
    }

    return adjusted_ptr;
  }
};

r2i::IPreprocessing *
FactoryMakePreprocessing () {
  return new Example ();
}
