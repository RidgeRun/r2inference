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

#include <r2i/r2i.h>

#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [Inception ONNX Model] "
            << "-s [Model Input Size] "
            << "-I [Preprocess Module] "
            << "-O [Postprocess Module] \n"
            << " Example: "
            << " ./inception -i cat.jpg -m graph_inceptionv2.onnx "
            << "-s 224"
            << std::endl;
}

std::shared_ptr<r2i::IFrame> LoadImage(const std::string &path, int req_width,
                                       int req_height,
                                       std::shared_ptr<r2i::IPreprocessing> preprocessing,
                                       std::shared_ptr<r2i::IFrame> in_frame,
                                       std::shared_ptr<r2i::IFrame> out_frame,
                                       r2i::RuntimeError &error) {
  int channels = 3;
  int width, height, cp;

  if (!in_frame) {
    error.Set (r2i::RuntimeError::Code::FILE_ERROR,
               "Null IFrame object");
    return nullptr;
  }

  if (!preprocessing) {
    error.Set (r2i::RuntimeError::Code::FILE_ERROR,
               "Null Preprocessing object");
    return nullptr;
  }

  unsigned char *img = stbi_load(path.c_str(), &width, &height, &cp, channels);
  if (!img) {
    error.Set (r2i::RuntimeError::Code::FILE_ERROR,
               "Error while loading the image file");
    std::cerr << "The picture " << path << " could not be loaded";
    return nullptr;
  }

  error = in_frame->Configure (img, width, height,
                               r2i::ImageFormat::Id::RGB);

  error = preprocessing->Apply(in_frame, out_frame);

  free (img);

  return out_frame;
}

bool ParseArgs (int &argc, char *argv[], std::string &image_path,
                std::string &model_path, int &index, int &size,
                std::string &preprocess_module, std::string &postprocess_module) {
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
        preprocess_module = optarg;
        break;
      case 'O' :
        postprocess_module = optarg;
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
  std::string preprocess_module;
  std::string postprocess_module;
  int Index = 0;
  int size = 0;

  if (false == ParseArgs (argc, argv, image_path, model_path, Index,
                          size, preprocess_module, postprocess_module)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty ()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   r2i::FrameworkCode::ONNXRT,
                   error);

  if (nullptr == factory) {
    std::cerr << "ONNXRT backend is not built: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading Model: " << model_path << std::endl;
  auto loader = factory->MakeLoader (error);
  std::shared_ptr<r2i::IModel> model = loader->Load (model_path, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::shared_ptr<r2i::IPreprocessing> preprocessing = loader->LoadPreprocessing(
        preprocess_module, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    loader->LoadPostprocessing(postprocess_module, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Setting model to engine" << std::endl;
  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine (error);
  error = engine->SetModel (model);

  std::cout << "Configuring ONNXRT session parameters" << std::endl;
  auto params = factory->MakeParameters (error);
  error = params->Configure(engine, model);
  /* Set OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING */
  error = params->Set ("logging-level", 2);
  error = params->Set ("log-id", "onnxrt_example");
  error = params->Set ("intra-num-threads", 1);
  /* Set GraphOptimizationLevel::ORT_ENABLE_EXTENDED */
  error = params->Set ("graph-optimization-level", 2);

  if (error.IsError ()) {
    std::cerr << "Parameters error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading image: " << image_path << std::endl;
  std::shared_ptr<r2i::IFrame> in_frame = factory->MakeFrame (error);
  std::shared_ptr<r2i::IFrame> out_frame = factory->MakeFrame (error);
  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[size * size * 3],
                                     std::default_delete<float[]>());
  error = out_frame->Configure (out_data.get(), size,
                                size,
                                r2i::ImageFormat::Id::RGB);
  out_frame = LoadImage (image_path, size, size, preprocessing, in_frame,
                         out_frame, error);
  if (error.IsError ()) {
    std::cerr << error.GetDescription() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Starting engine" << std::endl;
  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Engine start error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Predicting..." << std::endl;
  auto prediction = engine->Predict (out_frame, error);
  if (error.IsError ()) {
    std::cerr << "Engine prediction error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  /* Sort and print top prediction */
  postprocessing->Apply(prediction, error);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
