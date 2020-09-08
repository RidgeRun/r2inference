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

#ifndef R2I_PREDICTION_CLASSIFICATION_H
#define R2I_PREDICTION_CLASSIFICATION_H

#include <r2i/predictiondecorator.h>

namespace r2i {

class PredictionClassification: public PredictionDecorator {
 public:
  PredictionClassification();
  PredictionClassification(std::shared_ptr<IPrediction> base);
  ~PredictionClassification();
  RuntimeError SetScores(float *scores, unsigned int size);
  RuntimeError SetLabels(int *labels, unsigned int size);
  float *GetScores();
  int *GetLabels();

 private:
  std::vector<float> scores;
  std::vector<int> labels;
};

}


#endif // R2I_PREDICTION_CLASSIFICATION_H