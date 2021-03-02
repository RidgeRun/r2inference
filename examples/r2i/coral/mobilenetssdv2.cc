/* Copyright (C) 2018-2021 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include <algorithm>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <r2i/r2i.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/* Output signature of the architecture */
#define LOCATION 0
#define LABELS 1
#define SCORES 2
#define NUM_BOXES 3

#define NUM_LOCATION_PARAMS 4
#define OBJECT_THRESHOLD 0.4

void PrintBoxes (std::vector<std::shared_ptr<r2i::IPrediction>> predictions,
                 int width, int height) {
  r2i::RuntimeError error;

  /* The 4th tensor of shape [1] contains the number of detected boxes */
  int num_boxes = predictions[NUM_BOXES]->At(0, error);
  int cur_box = 0;
  int index = 0;
  int left = 0, top = 0, right = 0, bottom = 0;
  int label = 0;
  float prob = 0;

  for (int b = 0; b < num_boxes; b++) {

    if (predictions[SCORES]->At(b, error) >= OBJECT_THRESHOLD) {
      index = NUM_LOCATION_PARAMS * b;
      /* Compute box coordinates based on original resolution. The coordinates
      are normalized from 0 to 1. */
      top = predictions[LOCATION]->At(index, error) * height;
      left = predictions[LOCATION]->At(index + 1, error) * width;
      bottom = predictions[LOCATION]->At(index + 2, error) * height;
      right = predictions[LOCATION]->At(index + 3, error) * width;
      label = predictions[LABELS]->At(b, error);
      prob = predictions[SCORES]->At(b, error);

      std::cout << "Box " << cur_box << std::endl;
      std::cout << "\tTop-left corner: x=" << left << ", y=" << top
                << std::endl;
      std::cout << "\tBottom-right corner: x=" << right << ", y=" << bottom
                << std::endl;
      std::cout << "\tClass: " << label << ", prob=" << prob << std::endl;
      cur_box ++;
    }
  }
}

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [Inception TfLite Model] "
            << "-s [Model Input Size] "
            << "-I [Input Node] "
            << "-O [Output Node] \n"
            << " Example: "
            << " ./mobilenetssdv2 -i cat.jpg "
            << "-m ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite "
            << "-s 224"
            << std::endl;
}

std::unique_ptr<float[]> PreProcessImage (const unsigned char *input,
    int width, int height, int reqwidth, int reqheight) {

  const int channels = 3;
  const int scaled_size = channels * reqwidth * reqheight;
  std::unique_ptr<unsigned char[]> scaled (new unsigned char[scaled_size]);
  std::unique_ptr<float[]> adjusted (new float[scaled_size]);

  stbir_resize_uint8(input, width, height, 0, scaled.get(), reqwidth,
                     reqheight, 0, channels);

  for (int i = 0; i < scaled_size; i += channels) {
    /* RGB = (RGB - Mean)*StdDev */
    adjusted[i + 0] = (static_cast<float>(scaled[i + 0]) - 127.5) / 127.5;
    adjusted[i + 1] = (static_cast<float>(scaled[i + 1]) - 127.5) / 127.5;
    adjusted[i + 2] = (static_cast<float>(scaled[i + 2]) - 127.5) / 127.5;
  }

  return adjusted;
}

std::unique_ptr<float[]> LoadImage(const std::string &path, int reqwidth,
                                   int reqheight, int &width, int &height) {
  int channels = 3;
  int cp;

  unsigned char *img = stbi_load(path.c_str(), &width, &height, &cp, channels);
  if (!img) {
    std::cerr << "The picture " << path << " could not be loaded";
    return nullptr;
  }

  auto ret = PreProcessImage(img, width, height, reqwidth, reqheight);
  free (img);

  return ret;
}

bool ParseArgs (int &argc, char *argv[], std::string &image_path,
                std::string &model_path, int &index, int &size,
                std::string &in_node, std::string &out_node) {

  int option = 0;
  while ((option = getopt(argc, argv, "i:m:p:s:I:O:")) != -1) {
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
      case 's' :
        size = std::stoi (optarg);
        break;
      case 'I' :
        in_node = optarg;
        break;
      case 'O' :
        out_node = optarg;
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
  std::string in_node;
  std::string out_node;
  int Index = 0;
  int size = 0;
  int width = 0;
  int height = 0;

  if (false == ParseArgs (argc, argv, image_path, model_path, Index,
                          size, in_node, out_node)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty ()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   r2i::FrameworkCode::CORAL,
                   error);

  if (nullptr == factory) {
    std::cerr << "Coral from Google backend is not built: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading Model: " << model_path << std::endl;
  auto loader = factory->MakeLoader (error);
  std::shared_ptr<r2i::IModel> model = loader->Load (model_path, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Setting model to engine" << std::endl;
  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine (error);
  error = engine->SetModel (model);

  std::cout << "Loading image: " << image_path << std::endl;
  std::unique_ptr<float[]> image_data = LoadImage (image_path, size, size, width,
                                        height);

  std::cout << "Configuring frame" << std::endl;
  std::shared_ptr<r2i::IFrame> frame = factory->MakeFrame (error);

  error = frame->Configure (image_data.get(), size, size,
                            r2i::ImageFormat::Id::RGB);

  std::cout << "Starting engine" << std::endl;
  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Engine start error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Predicting..." << std::endl;
  std::vector<std::shared_ptr<r2i::IPrediction>> predictions;
  error = engine->Predict (frame, predictions);
  if (error.IsError ()) {
    std::cerr << "Engine prediction error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  PrintBoxes (predictions, width, height);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
