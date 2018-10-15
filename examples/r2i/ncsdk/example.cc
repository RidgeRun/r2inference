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
#include <CImg.h>

#define GOOGLENET_DIM 224

float GoogleNetMean[] = {0.40787054 * 255.0, 0.45752458 * 255.0, 0.48109378 * 255.0};

void print_usage() {
  printf("Usage: example -i [JPG input_image] -m [Model] \n");
}

std::shared_ptr<void> LoadImage (std::string image_path, int size,
                                 float *mean) {
  std::shared_ptr<unsigned char> data (nullptr);
  unsigned char *img_data;
  cimg_library::CImg<unsigned char> img(image_path.c_str());

  printf ("Original Image: Width: %d, Height: %d, Depth: %d, Channels: %d \n",
          img.width(), img.height(), img.depth(), img.spectrum());

  img.resize (GOOGLENET_DIM, GOOGLENET_DIM, true);

  printf ("Resized Image: Width: %d, Height: %d, Depth: %d, Channels: %d \n",
          img.width(), img.height(), img.depth(), img.spectrum());

  /* TODO: Here we need to apply the mean correction to the image */
  img_data = img.data();
  data = std::make_shared<unsigned char> (*img_data);
  return data;
}

int main (int argc, char *argv[]) {
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::ncsdk::Prediction> prediction;
  r2i::ncsdk::Engine engine;
  r2i::ncsdk::Loader loader;
  std::shared_ptr<r2i::ncsdk::Frame> frame (new r2i::ncsdk::Frame());
  std::shared_ptr<void> in_data;
  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  int option = 0;

  if (argc < 2) {
    print_usage();
    exit(EXIT_FAILURE);
  }
  while ((option = getopt(argc, argv, "i:m:")) != -1) {
    switch (option) {
      case 'i' : image_path = optarg;
        break;
      case 'm' : model_path  = optarg;
        break;
      default: print_usage();
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
  in_data = LoadImage (image_path, GOOGLENET_DIM, GoogleNetMean);

  printf("Configure Frame...\n");
  error = frame->Configure (in_data, GOOGLENET_DIM, GOOGLENET_DIM,
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

  float *out_data = reinterpret_cast<float *>(prediction->GetResultData());
  unsigned int out_size = prediction->GetResultSize();

  int numResults = out_size / (int)sizeof(float);

  printf("Result size was: %d, which represents %d elements \n", out_size,
         numResults);

  float maxResult = 0.0;
  int maxIndex = -1;
  for (int index = 0; index < numResults; index++) {
    if (out_data[index] > maxResult) {
      maxResult = out_data[index];
      maxIndex = index;
    }
  }
  printf("Index of top result is: %d\n", maxIndex);
  printf("Probability of top result is: %f\n", out_data[maxIndex]);


  printf("Stop Engine...\n");
  error = engine.Stop ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Stop Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }
  return 0;
}
