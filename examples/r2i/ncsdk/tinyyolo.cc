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
  double max_prob = 0;
  int max_prob_row = 0;
  int max_prob_col = 0;
  int max_prob_class = 0;
  int max_prob_box = 0;
  double box[4];

  int box_probs_start = GRID_H * GRID_W * CLASSES;
  int all_boxes_start = GRID_H * GRID_W * CLASSES + GRID_H * GRID_W * BOXES;
  int index;

  std::string labels [CLASSES] = {"aeroplane", "bicycle", "bird", "boat",
                                  "bottle", "bus", "car", "cat", "chair",
                                  "cow", "diningtable", "dog", "horse",
                                  "motorbike", "person", "pottedplant",
                                  "sheep", "sofa", "train", "tvmonitor"
                                 };

  /* Find the box index with the highest probability */
  for (i = 0; i < GRID_H; i++) {        /* Iterate rows    */
    for (j = 0; j < GRID_W; j++) {      /* Iterate columns */
      for (c = 0; c < CLASSES; c++) {   /* Iterate classes */
        index = (i * GRID_W + j) * CLASSES + c;
        class_prob = prediction->At (index, error);
        for (b = 0; b < BOXES; b++) {   /* Iterate boxes   */
          index = (i * GRID_W + j) * BOXES + b;
          box_prob = prediction->At (box_probs_start + index, error);
          prob = class_prob * box_prob;
          if (prob > PROB_THRESH) {
            max_prob = prob;
            max_prob_row = i;
            max_prob_col = j;
            max_prob_class = c;
            max_prob_box = b;
          }
        }
      }
    }
  }

  /* Convert box coordinates to pixels */
  /*
   * box position (x_center,y_center) is normalized inside each cell from 0 to 1
   * width and heigh are also normalized, but with image size as reference
   * box is ordered [x_center,y_center,width,height]
   * box dimmensions are squared on the ncappzoo python example
   */
  index = ((max_prob_row * GRID_W + max_prob_col) * BOXES + max_prob_box ) * 4;
  box[0] = prediction->At (all_boxes_start + index, error);
  box[1] = prediction->At (all_boxes_start + index + 1, error);
  box[2] = prediction->At (all_boxes_start + index + 2, error);
  box[3] = prediction->At (all_boxes_start + index + 3, error);

  /* adjust the box anchor according to its cell and grid dim */
  box[0] += max_prob_col;
  box[1] += max_prob_row;
  box[0] /= GRID_H;
  box[1] /= GRID_W;

  box[2] *= box[2];
  box[3] *= box[3];
  box[0] *= input_image_width;
  box[1] *= input_image_height;
  box[2] *= input_image_width;
  box[3] *= input_image_height;

  std::cout << "Box highest probaility:" ;
  std::cout << "[class:'" << labels[max_prob_class] << "', ";
  std::cout << "x_center:" << box[0] << ", ";
  std::cout << "y_center:" << box[1] << ", ";
  std::cout << "width:" << box[2] << ", ";
  std::cout << "height:" << box[3] << ", ";
  std::cout << "prob:" << max_prob << "]" << std::endl;

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
