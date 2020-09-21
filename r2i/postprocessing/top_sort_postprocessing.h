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

#ifndef R2I_TOP_SORT_POSTPROCESSING_H
#define R2I_TOP_SORT_POSTPROCESSING_H

#include <r2i/ipostprocessing.h>

namespace r2i {

class TopSortPostprocessing: public r2i::IPostprocessing {
 public:
  std::shared_ptr<r2i::IPrediction> Apply(
    std::shared_ptr<r2i::IPrediction>
    prediction,
    r2i::RuntimeError &error) override;

 private:
  std::shared_ptr<r2i::IPrediction> SortPrediction (
    std::shared_ptr<r2i::IPrediction>
    prediction, r2i::RuntimeError &error);
};

}  // namespace r2i

#endif  // R2I_TOP_SORT_POSTPROCESSING_H
