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

#ifndef R2I_PREDICTION_DETECTION_H
#define R2I_PREDICTION_DETECTION_H

#include <r2i/predictiondecorator.h>

namespace r2i {

struct BBox {
  double x;
  double y;
  double width;
  double height;
};


class PredictionDetection: public PredictionDecorator {
 public:
  PredictionDetection();
  PredictionDetection(std::shared_ptr<IPrediction> base);
  ~PredictionDetection();
  RuntimeError SetBoundingBoxes(BBox *bounding_boxes, unsigned int size);
  BBox *GetBoundingBoxes();

 private:
  std::vector<BBox> bounding_boxes;
};

}

#endif // R2I_PREDICTION_CLASSIFICATION_H
