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

#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <bits/stdc++.h>
#include <algorithm>

#include <r2i/r2i.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* Tiny YOLO oputput parameters */
/* Input image dim */
#define DIM 448
/* Grid dim */
#define GRID_H 7
#define GRID_W 7
/* Number of classes */
#define CLASSES 20
/* Number of boxes per cell */
#define BOXES 2
/* Probability threshold */
#define PROB_THRESH 0.07
/* Intersection over union threshold */
#define IOU_THRESH 0.35

struct box {
  std::string label;
  double x_center;
  double y_center;
  double width;
  double height;
  double prob;
};

void Box2Pixels (box *normalized_box, int row, int col, int image_width,
                 int image_height) {
  /* Convert box coordinates to pixels
   * box position (x_center,y_center) is normalized inside each cell from 0 to 1
   * width and heigh are also normalized, but with image size as reference
   * box is ordered [x_center,y_center,width,height]
   */
  /* adjust the box center according to its cell and grid dim */
  normalized_box->x_center += col;
  normalized_box->y_center += row;
  normalized_box->x_center /= GRID_H;
  normalized_box->y_center /= GRID_W;

  /* adjust the lengths and widths */
  normalized_box->width *= normalized_box->width;
  normalized_box->height *= normalized_box->height;

  /* scale the boxes to the image size in pixels */
  normalized_box->x_center *= image_width;
  normalized_box->y_center *= image_height;
  normalized_box->width *= image_width;
  normalized_box->height *= image_height;
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
  intersection_dim_1 = std::min(box_1.x_center + 0.5 * box_1.width,
                                box_2.x_center + 0.5 * box_2.width) -
                       std::max(box_1.x_center - 0.5 * box_1.width,
                                box_2.x_center - 0.5 * box_2.width);

  /* Second dimension of the intersecting box */
  intersection_dim_2 = std::min(box_1.y_center + 0.5 * box_1.height,
                                box_2.y_center + 0.5 * box_2.height) -
                       std::max(box_1.y_center - 0.5 * box_1.height,
                                box_2.y_center - 0.5 * box_2.height);

  if ((intersection_dim_1 < 0) || (intersection_dim_2 < 0)) {
    intersection_area = 0;
  } else {
    intersection_area =  intersection_dim_1 * intersection_dim_2;
  }
  union_area = box_1.width * box_1.height + box_2.width * box_2.height -
               intersection_area;
  return intersection_area / union_area;
}

void PrintBox (box in_box) {
  std::cout << "Box:" ;
  std::cout << "[class:'" << in_box.label << "', ";
  std::cout << "x_center:" << in_box.x_center << ", ";
  std::cout << "y_center:" << in_box.y_center << ", ";
  std::cout << "width:" << in_box.width << ", ";
  std::cout << "height:" << in_box.height << ", ";
  std::cout << "prob:" << in_box.prob << "]" << std::endl;
}

void PrintTopPrediction (std::shared_ptr<r2i::IPrediction> prediction,
                         int input_image_width, int input_image_height) {
  /*
   * Tiny yolo parameters:
   *    Grid: 7*7
   *    Boxes per grid cell: 2
   *    Number of classes: 20
   *    Classes: ["aeroplane", "bicycle", "bird", "boat", "bottle",
   *              "bus", "car", "cat", "chair", "cow", "diningtable",
   *              "dog", "horse", "motorbike", "person", "pottedplant",
   *              "sheep", "sofa", "train", "tvmonitor"]
   *
   * Prediction structure:
   *    [0:980]: 7*7*20 probability per class per grid cell
   *    [980:1078]: 7*7*2 probability multiplicator for each box in the grid
   *    [1078:1470]: 7*7*2*4 [x,y,w,h] for each box in the grid
   */
  int i, j, c, b;
  r2i::RuntimeError error;
  double class_prob;
  double box_prob;
  double prob;
  box result;

  int box_probs_start = GRID_H * GRID_W * CLASSES;
  int all_boxes_start = GRID_H * GRID_W * CLASSES + GRID_H * GRID_W * BOXES;
  int index;

  std::list<box> boxes;
  std::list<box>::iterator it;

  std::string labels [CLASSES] = {"aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse",
                                  "motorbike", "person", "pottedplant",
                                  "sheep", "sofa", "train", "tvmonitor"
                                 };

  for (i = 0; i < GRID_H; i++) {        /* Iterate rows    */
    for (j = 0; j < GRID_W; j++) {      /* Iterate columns */
      for (c = 0; c < CLASSES; c++) {   /* Iterate classes */
        index = (i * GRID_W + j) * CLASSES + c;
        class_prob = prediction->At (index, error);
        for (b = 0; b < BOXES; b++) {   /* Iterate boxes   */
          index = (i * GRID_W + j) * BOXES + b;
          box_prob = prediction->At (box_probs_start + index, error);
          prob = class_prob * box_prob;
          /* If the probability is over the threshold add it to the boxes list */
          if (prob > PROB_THRESH) {
            index = ((i * GRID_W + j) * BOXES + b ) * 4;
            result.label = labels[c];
            result.x_center = prediction->At (all_boxes_start + index, error);
            result.y_center = prediction->At (all_boxes_start + index + 1, error);
            result.width = prediction->At (all_boxes_start + index + 2, error);
            result.height = prediction->At (all_boxes_start + index + 3, error);
            result.prob = prob;
            Box2Pixels(&result, i, j, input_image_width, input_image_height);
            boxes.push_front(result);
          }
        }
      }
    }
  }

  /* Remove duplicated boxes. A box is considered a duplicate if its
   * intersection over union metric is above a threshold
   */
  IntersectionOverUnion (boxes.front(), boxes.back());


  /* Print all resulting boxes */
  for (it = boxes.begin(); it != boxes.end(); ++it)
    PrintBox(*it);

}

void PrintUsage() {
  std::cerr << "Usage: example -i [JPG input_image] -m [TinyYOLO Model]" <<
            std::endl;
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
    adjusted[i + 2] = static_cast<float>(scaled[i + 2]) / 255.0;
    adjusted[i + 1] = static_cast<float>(scaled[i + 1]) / 255.0;
    adjusted[i + 0] = static_cast<float>(scaled[i + 0]) / 255.0;
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

  std::cout << "Loading Model: " << model_path << "..." << std::endl;
  auto loader = factory->MakeLoader (error);
  auto model = loader->Load (model_path, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Setting model to engine..." << std::endl;
  auto engine = factory->MakeEngine (error);
  error = engine->SetModel (model);

  std::cout << "Loading image: " << image_path << "..." << std::endl;
  std::unique_ptr<float[]> image_data = LoadImage (image_path, DIM, DIM, &width,
                                        &height);

  std::cout << "Configuring frame..." << std::endl;
  std::shared_ptr<r2i::IFrame> frame = factory->MakeFrame (error);
  error = frame->Configure (image_data.get(), DIM, DIM,
                            r2i::ImageFormat::Id::RGB);

  std::cout << "Starting engine..." << std::endl;
  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Engine start error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Predicting..." << std::endl;
  auto prediction = engine->Predict (frame, error);
  if (error.IsError ()) {
    std::cerr << "Engine prediction error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  PrintTopPrediction (prediction, width, height);

  std::cout << "Stopping engine..." << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
