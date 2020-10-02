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

#include "r2i/postprocessing/parse_bounding_boxes_tinyyolov2.h"

/*Definition of Euler's number (M_E) constant */
#include <cmath>

#define GRID_H 13
#define GRID_W 13
#define BOXES_SIZE 5
#define BOX_DIM 5
#define CLASSES 20
#define OBJ_THRESHOLD (-2.5)
#define CLASS_THRESHOLD 5

namespace r2i {

static const float box_anchors[] =
{ 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

static double
sigmoid (double x) {
  return 1.0 / (1.0 + pow (M_E, -1.0 * x));
}

RuntimeError ParseBoundingBoxesTinyyolov2::Apply(
  std::vector< std::shared_ptr<r2i::IPrediction> > &predictions,
  std::vector< std::shared_ptr<InferenceOutput> > &outputs) {
  float *prediction_data;
  unsigned int prediction_data_size;
  unsigned int num_predictions;
  r2i::RuntimeError error;
  std::shared_ptr<r2i::Detection> detection;
  std::vector< r2i::DetectionInstance > boxes;

  /* Number of predictions */
  num_predictions = predictions.size();

  for (unsigned int i = 0; i < num_predictions; ++i) {

    detection = std::make_shared<r2i::Classification>();
    if (!detection) {
      error.Set (RuntimeError::Code::MODULE_ERROR,
                 "Failed to create Detection instance");
      return error;
    }

    ParseBoxes(detection);
  }

  return error;
}

void
ParseBoundingBoxesTinyyolov2::BoxToPixels (BBox *normalized_box, int row,
    int col, int box) {
  int grid_size = 32;
  r2i::RuntimeError error;

  if (!normalized_box)
    return;

  /* adjust the box center according to its cell and grid dim */
  normalized_box->x = (col + sigmoid (normalized_box->x)) * grid_size;
  normalized_box->y = (row + sigmoid (normalized_box->y)) * grid_size;

  /* adjust the lengths and widths */
  normalized_box->width =
    pow (M_E, normalized_box->width) * box_anchors[2 * box] * grid_size;
  normalized_box->height =
    pow (M_E, normalized_box->height) * box_anchors[2 * box + 1] * grid_size;
}

RuntimeError
ParseBoundingBoxesTinyyolov2::ParseBoxes(std::shared_ptr<r2i::Detection>
    detection) {
  float *network_output;
  int i, j, b, c;

  if (!detection)
    return;

  network_output = static_cast<float *>(prediction->GetResultData());

  for (i = 0; i < GRID_H; i++) {
    for (j = 0; j < GRID_W; j++) {
      for (b = 0; b < BOXES_SIZE; b++) {
        int index;
        double obj_prob;
        double cur_class_prob, max_class_prob;
        int max_class_prob_index;

        index = ((i * GRID_W + j) * BOXES_SIZE + b) * (BOX_DIM + CLASSES);
        obj_prob = network_output[index + 4];


        if (obj_prob > OBJ_THRESHOLD) {

          max_class_prob = 0;
          max_class_prob_index = 0;
          for (c = 0; c < CLASSES; c++) {
            cur_class_prob = network_output[index + BOX_DIM + c];
            if (cur_class_prob > max_class_prob) {
              max_class_prob = cur_class_prob;
              max_class_prob_index = c;
            }
          }
          if (max_class_prob > CLASS_THRESHOLD) {
            BBox result;
            result.label = max_class_prob_index;
            result.prob = max_class_prob;
            result.x = network_output[index];
            result.y = network_output[index + 1];
            result.width = network_output[index + 2];
            result.height = network_output[index + 3];
            BoxToPixels (&result, i, j, b);
            result.x = result.x - result.width * 0.5;
            result.y = result.y - result.height * 0.5;

            //~ printf("prob: %.2f\t label: %d\n", max_class_prob, result.label);
            //~ printf("x: %d\ty: %d\t width: %d\t height: %d\n", (int)result.x,
            //~ (int)result.y, (int)result.width, (int)result.height);
          }
        }
      }
    }
  }
}

}  // namespace r2i

r2i::IPostprocessing *
FactoryMakePostprocessing () {
  return new r2i::ParseBoundingBoxesTinyyolov2 ();
}
