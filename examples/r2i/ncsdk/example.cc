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


#define GOOGLENET_DIM 224

float GoogleNetMean[] = {0.40787054 * 255.0, 0.45752458 * 255.0, 0.48109378 * 255.0};

void print_usage() {
  printf("Usage: example -i [JPG input_image] -m [Model] \n");
}


float *LoadImage(const char *path, int reqsize, float *mean) {
  int width, height, cp, i;
  unsigned char *img, *imgresized;
  float *imgfp32;
  unsigned int imageSize;

  img = stbi_load(path, &width, &height, &cp, 3);
  if (!img) {
    printf("The picture %s could not be loaded\n", path);
    return 0;
  }
  imgresized = (unsigned char *) malloc(3 * reqsize * reqsize);
  if (!imgresized) {
    free(img);
    perror("malloc");
    return 0;
  }
  stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
  free(img);

  imageSize = sizeof(*imgfp32) * reqsize * reqsize * 3;
  imgfp32 = (float *) malloc(imageSize);

  printf("size: %i\n", imageSize);
  if (!imgfp32) {
    free(imgresized);
    perror("malloc");
    return 0;
  }
  for (i = 0; i < reqsize * reqsize * 3; i++)
    imgfp32[i] = imgresized[i];
  free(imgresized);
  for (i = 0; i < reqsize * reqsize; i++) {
    float blue, green, red;
    blue = imgfp32[3 * i + 2];
    green = imgfp32[3 * i + 1];
    red = imgfp32[3 * i + 0];

    imgfp32[3 * i + 0] = blue - mean[0];
    imgfp32[3 * i + 1] = green - mean[1];
    imgfp32[3 * i + 2] = red - mean[2];

  }
  return imgfp32;
}



int main (int argc, char *argv[]) {
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::ncsdk::Prediction> prediction;
  r2i::ncsdk::Engine engine;
  r2i::ncsdk::Loader loader;
  std::shared_ptr<r2i::ncsdk::Frame> frame (new r2i::ncsdk::Frame());
  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  unsigned int Index = 0;
  int option = 0;
  float *image_data;

  if (argc < 2) {
    print_usage();
    exit(EXIT_FAILURE);
  }
  while ((option = getopt(argc, argv, "i:m:p:")) != -1) {
    switch (option) {
      case 'i' :
        image_path = optarg;
        break;
      case 'm' :
        model_path  = optarg;
        break;
      case 'p' :
        Index  = atoi(optarg);
        break;

      default:
        print_usage();
        exit(EXIT_FAILURE);
    }
  }

  printf("Loading Model: %s ...\n", model_path.c_str());
  model = loader.Load (model_path, error);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Loader Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  printf("Setting Model to Engine...\n");
  error = engine.SetModel (model);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine SetModel Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  printf("Process Image: %s...\n", image_path.c_str());
  image_data = LoadImage (image_path.c_str(), GOOGLENET_DIM, GoogleNetMean);

  std::shared_ptr<void> in_data = std::make_shared<float *>(image_data);

  printf("Configure Frame...\n");
  error = frame->Configure (image_data, GOOGLENET_DIM, GOOGLENET_DIM,
                            r2i::ImageFormat::Id::RGB);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Frame Configuration Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  printf("Start Engine...\n");
  error = engine.Start ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Start Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  printf("Predict...\n");
  prediction =
    std::dynamic_pointer_cast<r2i::ncsdk::Prediction, r2i::IPrediction>
    (engine.Predict (frame, error));
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Prediction Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  unsigned int out_size = prediction->GetResultSize();

  int numResults = out_size / (int)sizeof(float);

  printf("Result size was: %d, which represents %d elements \n", out_size,
         numResults);


  double Result = 0.0;

  Result = prediction->At(Index, error);
  printf("Probability at index %i  is: %lf\n", Index, Result);


  printf("Stop Engine...\n");
  error = engine.Stop ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Stop Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }
  return 0;
}
