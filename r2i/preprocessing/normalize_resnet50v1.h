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

#ifndef R2I_NORMALIZE_RESNET50V1_PREPROCESSING_H
#define R2I_NORMALIZE_RESNET50V1_PREPROCESSING_H

#include <r2i/preprocessing/normalize.h>

namespace r2i {

class NormalizeResnet50V1: public r2i::Normalize {
 public:
  NormalizeResnet50V1 ();
 private:
  r2i::RuntimeError SetNormalizationParameters () override;
};

}  // namespace r2i

#endif  // R2I_NORMALIZE_RESNET50V1_PREPROCESSING_H
