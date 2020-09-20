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

#ifndef R2I_MEAN_STD_PREPROCESSING_H
#define R2I_MEAN_STD_PREPROCESSING_H

#include <r2i/ipreprocessing.h>

namespace r2i {

class MeanStdPreprocessing: public r2i::IPreprocessing {
 public:
   MeanStdPreprocessing ();
   r2i::RuntimeError Apply(std::shared_ptr<r2i::IFrame> in_frame,
                          std::shared_ptr<r2i::IFrame> out_frame, int required_width, int required_height,
                          r2i::ImageFormat::Id required_format_id) override;
   std::vector<r2i::ImageFormat> GetAvailableFormats() override;
   std::vector<std::tuple<int, int>> GetAvailableDataSizes() override;

 private:
   std::shared_ptr<float> processed_data;
   std::vector<std::tuple<int, int>> dimensions;
   std::vector<r2i::ImageFormat> formats;

   r2i::RuntimeError Validate (int required_width, int required_height,
                             r2i::ImageFormat::Id required_format_id);
   std::shared_ptr<float> PreProcessImage (const unsigned char *input,
                                           int width, int height, int required_width, int required_height);
   r2i::IPreprocessing *FactoryMakePreprocessing ();
};

}  // namespace r2i

#endif  // R2I_MEAN_STD_PREPROCESSING_H
