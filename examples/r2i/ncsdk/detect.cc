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
#include <iostream>
#include <getopt.h>
#include <string>
#include <r2i/r2i.h>
#include <r2i/ncsdk/engine.h>
#include <r2i/ncsdk/loader.h>
#include <r2i/ncsdk/model.h>
#include <r2i/ncsdk/frame.h>
#include <r2i/ncsdk/prediction.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* Tiny YOLO oputput parameters */
#define DIM 448
#define GRID_H 7
#define GRID_W 7
#define CLASSES 20
#define BOXES 2
#define THRESHOLD 0.07


void
print_usage () {
  printf ("Usage: example -i [JPG input_image] -m [Model] \n");
}


float *
LoadImage (const char *path, int reqsize, int &width, int &height) {
  int cp, i;
  unsigned char *img, *imgresized;
  float *imgfp32;
  unsigned int imageSize;

  img = stbi_load (path, &width, &height, &cp, 3);
  if (!img) {
    printf ("The picture %s could not be loaded\n", path);
    return 0;
  }
  imgresized = (unsigned char *) malloc (3 * reqsize * reqsize);
  if (!imgresized) {
    free (img);
    perror ("malloc");
    return 0;
  }
  stbir_resize_uint8 (img, width, height, 0, imgresized, reqsize, reqsize, 0,
                      3);
  free (img);

  imageSize = sizeof (*imgfp32) * reqsize * reqsize * 3;
  imgfp32 = (float *) malloc (imageSize);

  printf ("size: %i\n", imageSize);
  if (!imgfp32) {
    free (imgresized);
    perror ("malloc");
    return 0;
  }
  for (i = 0; i < reqsize * reqsize * 3; i++)
    imgfp32[i] = imgresized[i];
  free (imgresized);
  for (i = 0; i < reqsize * reqsize; i++) {
    imgfp32[3 * i + 0] = imgfp32[3 * i + 0] / 255;
    imgfp32[3 * i + 1] = imgfp32[3 * i + 1] / 255;
    imgfp32[3 * i + 2] = imgfp32[3 * i + 2] / 255;

  }
  return imgfp32;
}


bool
interpret_prediction (std::shared_ptr < r2i::ncsdk::Prediction > prediction,
                      int input_image_width, int input_image_height, double *box) {
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

  /* Find the box index with the highest probability */
  for (i = 0; i < GRID_H; i++) {        /* Iterate rows    */
    for (j = 0; j < GRID_W; j++) {      /* Iterate columns */
      for (c = 0; c < CLASSES; c++) {   /* Iterate classes */
        for (b = 0; b < BOXES; b++) {   /* Iterate boxes   */
          class_prob = prediction->At ((i * GRID_W + j) * CLASSES + c, error);
          box_prob =
            prediction->At (GRID_H * GRID_W * CLASSES + (i * GRID_W + j) * BOXES + b,
                            error);
          prob = class_prob * box_prob;
          if (prob > max_prob) {
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

  if (max_prob < THRESHOLD) {
    return false;
  }

  /* Convert box coordinates to pixels */
  /*
   * box position (x,y) is normalized inside each cell from 0 to 1
   * width and heigh are also normalized, but with image size as reference
   * box is ordered [x,y,width,height]
   * box dimmensions are squared on the ncappzoo python example
   */
  for (i = 0; i < 4; i++) {
    box[i] =
      prediction->At (GRID_H * GRID_W * CLASSES + GRID_H * GRID_W * BOXES +
                      max_prob_row * max_prob_col * max_prob_box + i, error);
  }
  box[0] += max_prob_col;
  box[1] += max_prob_row;
  box[0] /= (DIM / GRID_H);
  box[1] /= (DIM / GRID_W);
  box[2] *= box[2];
  box[3] *= box[3];
  box[0] *= input_image_width;
  box[1] *= input_image_height;
  box[2] *= input_image_width;
  box[3] *= input_image_height;

  /* Set the class and probability */
  box[4] = (double) max_prob_class;
  box[5] = max_prob;

  return true;
}

int
main (int argc, char *argv[]) {
  std::shared_ptr < r2i::IModel > model;
  std::shared_ptr < r2i::ncsdk::Prediction > prediction;
  r2i::ncsdk::Engine engine;
  r2i::ncsdk::Loader loader;
  std::shared_ptr < r2i::ncsdk::Frame > frame (new r2i::ncsdk::Frame ());
  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  int option = 0;
  float *image_data;
  int width, height;

  if (argc < 2) {
    print_usage ();
    exit (EXIT_FAILURE);
  }
  while ((option = getopt (argc, argv, "i:m:")) != -1) {
    switch (option) {
      case 'i':
        image_path = optarg;
        break;
      case 'm':
        model_path = optarg;
        break;

      default:
        print_usage ();
        exit (EXIT_FAILURE);
    }
  }

  printf ("Loading Model: %s ...\n", model_path.c_str ());
  model = loader.Load (model_path, error);
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Loader Error: %s\n", error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }

  printf ("Setting Model to Engine...\n");
  error = engine.SetModel (model);
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Engine SetModel Error: %s\n", error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }

  printf ("Process Image: %s...\n", image_path.c_str ());
  image_data = LoadImage (image_path.c_str (), DIM, width, height);

  std::shared_ptr < void >in_data = std::make_shared < float *>(image_data);

  printf ("Configure Frame...\n");
  error = frame->Configure (image_data, DIM, DIM, r2i::ImageFormat::Id::RGB);
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Frame Configuration Error: %s\n",
            error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }

  printf ("Start Engine...\n");
  error = engine.Start ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Engine Start Error: %s\n", error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }

  printf ("Predict...\n");
  prediction =
    std::dynamic_pointer_cast < r2i::ncsdk::Prediction, r2i::IPrediction >
    (engine.Predict (frame, error));
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Engine Prediction Error: %s\n", error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }

  unsigned int out_size = prediction->GetResultSize ();

  int numResults = out_size / (int) sizeof (float);

  printf ("Result size was: %d, which represents %d elements \n", out_size,
          numResults);


  double Result[6];
  /* interpret prediction outputs the hightest probability box that surpasses a threshold */
  if (interpret_prediction (prediction, width, height, Result)) {
    printf
    ("Box highest probaility: [x:%lf, y:%lf, width:%lf, height:%lf, class:%lf, prob:%lf] \n",
     Result[0], Result[1], Result[2], Result[3], Result[4], Result[5]);
  }

  printf ("Stop Engine...\n");
  error = engine.Stop ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
    printf ("Engine Stop Error: %s\n", error.GetDescription ().c_str ());
    exit (EXIT_FAILURE);
  }
  return 0;
}
