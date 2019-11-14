/* Copyright (C) 2019 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <bits/stdc++.h>
#include <algorithm>
#include <math.h>
#include <r2i/r2i.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* Tiny YOLOV2 oputput parameters */
/* Input image dim */
#define DIM 416
/* Grid dim */
#define GRID_H 13
#define GRID_W 13
/* Number of classes */
#define CLASSES 20
/* Number of boxes per cell */
#define BOXES 5
/* Box dim */
#define BOX_DIM 5
/* Probability threshold */
#define PROB_THRESH 0.08
/* Intersection over union threshold */
#define IOU_THRESH 0.30
/* Objectness threshold */
#define OBJ_THRESH 0.08
/* Grid cell size in pixels */
#define GRID_SIZE 32

const float box_anchors[] =
{ 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

struct box {
  std::string label;
  double x;
  double y;
  double width;
  double height;
  double prob;
};

/* sigmoid approximation as a lineal function */
static double
sigmoid (double x) {
  return 1.0 / (1.0 + pow (M_E, -1.0 * x));
}

void Box2Pixels (box *normalized_box, int row, int col, int box) {
  /* Convert box coordinates to pixels
   * box position (x,y) is normalized inside each cell from 0 to 1
   * width and heigh are also normalized, but with image size as reference
   * box is ordered [x_center,y_center,width,height]
   */
  /* Adjust the box center according to its cell and grid dim */

  normalized_box->x = (col + sigmoid (normalized_box->x)) * GRID_SIZE;
  normalized_box->y = (row + sigmoid (normalized_box->y)) * GRID_SIZE;

  normalized_box->width =
    pow (M_E, normalized_box->width) * box_anchors[2 * box] * GRID_SIZE;
  normalized_box->height =
    pow (M_E, normalized_box->height) * box_anchors[2 * box + 1] * GRID_SIZE;

}

void GetBoxesFromPrediction(std::shared_ptr<r2i::IPrediction> prediction,
                            int input_image_width, int input_image_height,
                            std::list<box> &boxes) {
  /*
   * Get all the boxes from the prediction and store them in a list
   * Tiny yolo parameters:
   *    Grid: 13*13
   *    Boxes per grid cell: 5
   *    Number of classes: 20
   *    Classes: ["aeroplane", "bicycle", "bird", "boat", "bottle",
   *              "bus", "car", "cat", "chair", "cow", "diningtable",
   *              "dog", "horse", "motorbike", "person", "pottedplant",
   *              "sheep", "sofa", "train", "tvmonitor"]
   *
   * Prediction structure:
   *    13*13*(5+20) [x,y,w,h,objectness] for each box in the grid
   */
  r2i::RuntimeError error;
  int i, j, c, b;
  int index;
  double max_class_prob;
  double cur_class_prob;
  double obj_prob;
  int max_class_prob_index;

  /*
   * This label list is highly dependent on the way the model was trained.
   * We are assumig the same labels that are used on the ncappzoo tinyyolov2
   * example.
   */
  std::string labels [CLASSES] = {"aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse",
                                  "motorbike", "person", "pottedplant",
                                  "sheep", "sofa", "train", "tvmonitor"
                                 };

  for (i = 0; i < GRID_H; i++) {        /* Iterate rows    */
    for (j = 0; j < GRID_W; j++) {      /* Iterate columns */
      for (b = 0; b < BOXES; b++) {   /* Iterate boxes   */
        index = ((i * GRID_W + j) * BOXES + b) * (BOX_DIM + CLASSES);
        obj_prob = prediction->At (index + 4, error);
        /* If the probability is over the threshold add it to the boxes list */
        if (obj_prob > OBJ_THRESH) {
          max_class_prob = 0;
          max_class_prob_index = 0;
          for (c = 0; c < CLASSES; c++) {
            cur_class_prob = prediction->At (index + BOX_DIM + c, error);
            if (cur_class_prob > max_class_prob) {
              max_class_prob = cur_class_prob;
              max_class_prob_index = c;
            }
          }

          if (max_class_prob > PROB_THRESH) {
            box result;
            result.label = labels[max_class_prob_index];
            result.x = prediction->At (index, error);
            result.y = prediction->At (index + 1, error);
            result.width = prediction->At (index + 2, error);
            result.height = prediction->At (index + 3, error);
            result.prob = max_class_prob;
            Box2Pixels(&result, i, j, b);
            result.x = result.x - result.width * 0.5;
            result.y = result.y - result.height * 0.5;
            boxes.push_front(result);
          }
        }
      }
    }
  }
}

double IntersectionOverUnion(box box_1, box box_2) {
  /*
   * Evaluate the intersection-over-union for two boxes
   * The intersection-over-union metric determines how close
   * two boxes are to being the same box.
   */
  double intersection_dim_1;
  double intersection_dim_2;
  double intersection_area;
  double union_area;

  /* First diminsion of the intersecting box */
  intersection_dim_1 = std::min(box_1.x + box_1.width,
                                box_2.x + box_2.width) -
                       std::max(box_1.x, box_2.x);

  /* Second dimension of the intersecting box */
  intersection_dim_2 = std::min(box_1.y + box_1.height,
                                box_2.y + box_2.height) -
                       std::max(box_1.y, box_2.y);

  if ((intersection_dim_1 < 0) || (intersection_dim_2 < 0)) {
    intersection_area = 0;
  } else {
    intersection_area =  intersection_dim_1 * intersection_dim_2;
  }

  union_area = box_1.width * box_1.height + box_2.width * box_2.height -
               intersection_area;
  return intersection_area / union_area;
}

void RemoveDuplicatedBoxes(std::list<box> &boxes) {
  /* Remove duplicated boxes. A box is considered a duplicate if its
   * intersection over union metric is above a threshold
   */
  double iou;
  std::list<box>::iterator it1;
  std::list<box>::iterator it2;

  for (it1 = boxes.begin(); it1 != boxes.end(); it1++) {
    for (it2 = std::next(it1); it2 != boxes.end(); it2++) {
      if (it1->label == it2->label) {
        iou = IntersectionOverUnion(*it1, *it2);
        if (iou > IOU_THRESH) {
          if (it1->prob > it2->prob) {
            boxes.erase(it2--);
          } else {
            boxes.erase(it1--);
            break;
          }
        }
      }
    }
  }
}

void PrintBox (box in_box) {
  std::cout << "Box:" ;
  std::cout << "[class:'" << in_box.label << "', ";
  std::cout << "x:" << in_box.x << ", ";
  std::cout << "y:" << in_box.y << ", ";
  std::cout << "width:" << in_box.width << ", ";
  std::cout << "height:" << in_box.height << ", ";
  std::cout << "prob:" << in_box.prob << "]" << std::endl;
}

void PrintTopPredictions (std::shared_ptr<r2i::IPrediction> prediction,
                          int input_image_width, int input_image_height) {
  /*
   * Print al boxes that surpass a probability threshold (PROB_THRESH).
   * Clustering is performed to remove duplicated boxes based on the
   * intersection over union metric.
   */
  std::list<box> boxes;

  GetBoxesFromPrediction(prediction, input_image_width, input_image_height,
                         boxes);

  RemoveDuplicatedBoxes(boxes);

  /* Print all resulting boxes */
  for (box b : boxes) {
    PrintBox (b);
  }
}

void PrintUsage() {
  std::cerr <<
            "Usage: example -i [JPG input_image] -m [TinyYOLOV2 NCSDK Model] \n"
            << "Example: ./tinyyolov2 -i dog.jpg -m graph_tinyyolov2_ncsdk"
            << std::endl;
}

std::unique_ptr<float[]> PreProcessImage (const unsigned char *input, int width,
    int height, int reqwidth, int reqheight) {
  const int channels = 3;
  const int scaled_size = channels * reqwidth * reqheight;

  std::unique_ptr<unsigned char[]> scaled (new unsigned char[scaled_size]);
  std::unique_ptr<float[]> adjusted (new float[scaled_size]);

  stbir_resize_uint8(input, width, height, 0, scaled.get(), reqwidth,
                     reqheight, 0, channels);

  for (int i = 0; i < scaled_size; i += channels) {
    adjusted[i + 0] = static_cast<float>(scaled[i + 0]) / 255;
    adjusted[i + 1] = static_cast<float>(scaled[i + 1]) / 255;
    adjusted[i + 2] = static_cast<float>(scaled[i + 2]) / 255;
  }

  return adjusted;
}

std::unique_ptr<float[]> LoadImage(const std::string &path, int reqwidth,
                                   int reqheight, int *width, int *height) {
  int channels = 3;
  int cp;

  unsigned char *img = stbi_load(path.c_str(), width, height, &cp, channels);
  if (!img) {
    std::cerr << "The picture " << path << " could not be loaded";
    return nullptr;
  }

  auto ret = PreProcessImage(img, *width, *height, reqwidth, reqheight);
  free (img);

  return ret;
}

bool ParseArgs (int &argc, char *argv[], std::string &image_path,
                std::string &model_path, int &index) {
  int option = 0;

  while ((option = getopt(argc, argv, "i:m:p:")) != -1) {
    switch (option) {
      case 'i' :
        image_path = optarg;
        break;
      case 'm' :
        model_path  = optarg;
        break;
      case 'p' :
        index  = std::stoi (optarg);
        break;
      default:
        return false;
    }
  }

  return true;
}


int main (int argc, char *argv[]) {
  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  int Index = 0;
  int width, height;

  if (false == ParseArgs (argc, argv, image_path, model_path, Index)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty ()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(r2i::FrameworkCode::NCSDK,
                 error);

  if (nullptr == factory) {
    std::cerr << "NCSDK backend is not built: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading Model: " << model_path << std::endl;
  auto loader = factory->MakeLoader (error);
  auto model = loader->Load (model_path, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Setting model to engine" << std::endl;
  auto engine = factory->MakeEngine (error);
  error = engine->SetModel (model);

  std::cout << "Loading image: " << image_path << std::endl;
  std::unique_ptr<float[]> image_data = LoadImage (image_path, DIM, DIM,
                                        &width, &height);

  std::cout << "Configuring frame" << std::endl;
  std::shared_ptr<r2i::IFrame> frame = factory->MakeFrame (error);
  error = frame->Configure (image_data.get(), DIM, DIM,
                            r2i::ImageFormat::Id::RGB);

  std::cout << "Starting engine" << std::endl;
  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Engine start error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Predicting" << std::endl;
  auto prediction = engine->Predict (frame, error);
  if (error.IsError ()) {
    std::cerr << "Engine prediction error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  PrintTopPredictions (prediction, width, height);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
