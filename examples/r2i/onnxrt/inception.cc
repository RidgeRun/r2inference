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

#include <r2i/classification.h>
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

r2i::RuntimeError LoadImage(const std::string &path, int req_width,
                            int req_height,
                            std::shared_ptr<r2i::IPreprocessing> preprocessing,
                            std::shared_ptr<r2i::IFrame> in_frame,
                            std::shared_ptr<r2i::IFrame> out_frame) {
  int channels = 3;
  int width = 0;
  int height = 0;
  int channels_in_file = 0;
  int required_width = 0;
  int required_height = 0;
  int required_channels = 0;
  unsigned char *scaled = nullptr;
  r2i::ImageFormat output_image_format;
  r2i::ImageFormat::Id input_image_format_id;
  r2i::DataType::Id input_image_datatype_id;
  r2i::RuntimeError error;
  std::shared_ptr<unsigned char> scaled_ptr;

  if (!in_frame) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Null IFrame object");
    return error;
  }

  if (!preprocessing) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Null Preprocessing object");
    return error;
  }

  unsigned char *img = stbi_load(path.c_str(), &width, &height, &channels_in_file,
                                 channels);
  if (!img) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Error while loading the image file");
    std::cerr << "The picture " << path << " could not be loaded";
    return error;
  }

  input_image_datatype_id = r2i::DataType::Id::UINT8;

  if (channels_in_file == 3) {
    input_image_format_id = r2i::ImageFormat::Id::RGB;
  } else {
    input_image_format_id = r2i::ImageFormat::Id::UNKNOWN_FORMAT;
  }

  required_width = out_frame->GetWidth();
  required_height = out_frame->GetHeight();
  required_channels = out_frame->GetFormat().GetNumPlanes();

  scaled_ptr = std::shared_ptr<unsigned char>(new unsigned char[required_width
               * required_height * required_channels],
               std::default_delete<const unsigned char[]>());
  scaled = scaled_ptr.get();

  stbir_resize_uint8(img, width, height, 0, scaled, required_width,
                     required_height, 0, required_channels);

  error = in_frame->Configure (scaled, required_width, required_height,
                               input_image_format_id, input_image_datatype_id);

  error = preprocessing->Apply(in_frame, out_frame);

  free (img);

  return error;
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
  double max = 0;
  int Index = 0;
  int size = 0;
  int max_index = 0;
  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  std::string preprocess_module;
  std::string postprocess_module;
  std::vector<std::shared_ptr<r2i::InferenceOutput>> outputs;
  std::vector<std::shared_ptr<r2i::IPrediction>> predictions;

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
                                r2i::ImageFormat::Id::RGB, r2i::DataType::Id::FLOAT);
  error = LoadImage (image_path, size, size, preprocessing, in_frame,
                     out_frame);
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
  /* Inception models just give one output */
  auto prediction = engine->Predict (out_frame, error);
  predictions.push_back(prediction);
  if (error.IsError ()) {
    std::cerr << "Engine prediction error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Postprocessing..." << std::endl;
  /* Sort and print top prediction */
  error = postprocessing->Apply(predictions, outputs);
  if (error.IsError ()) {
    std::cerr << "Postprocessing error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }
  /* Top score is at the beginning */
  auto output =
    std::dynamic_pointer_cast<r2i::Classification, r2i::InferenceOutput>
    (outputs.at(0));
  if (!output) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided output is not a Classification type");
    std::cerr << "Postprocessing error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::tuple<int, double> label = output->GetLabels().at(0);
  max_index = std::get<0>(label);
  max = std::get<1>(label);
  std::cout << "Highest probability is label "
            << max_index << " (" << max << ")" << std::endl;

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
