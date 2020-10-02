/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <r2i/r2i.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void PrintTopPrediction (std::shared_ptr<r2i::IPrediction> prediction) {
  r2i::RuntimeError error;
  int index = 0;
  double max = -1;
  int num_labels = prediction->GetResultSize() / sizeof(float);

  for (int i = 0; i < num_labels; ++i) {
    double current = prediction->At(i, error);
    if (current > max) {
      max = current;
      index = i;
    }
  }

  std::cout << "Highest probability is label "
            << index << " (" << max << ")" << std::endl;
}

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [Inception TfLite Model] "
            << "-s [Model Input Size] "
            << "-I [Input Node] "
            << "-O [Output Node] \n"
            << " Example: "
            << " ./inception -i cat.jpg -m graph_inceptionv2_tensorflow.pb "
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
                                   int reqheight) {
  int channels = 3;
  int width, height, cp;

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
                   r2i::FrameworkCode::TFLITE,
                   error);

  if (nullptr == factory) {
    std::cerr << "TensorFlow backend is not built: " << error << std::endl;
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
  std::unique_ptr<float[]> image_data = LoadImage (image_path, size,
                                        size);

  std::cout << "Configuring frame" << std::endl;
  std::shared_ptr<r2i::IFrame> frame = factory->MakeFrame (error);

  error = frame->Configure (image_data.get(), size, size,
                            r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);

  std::cout << "Starting engine" << std::endl;
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

  PrintTopPrediction (prediction);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
