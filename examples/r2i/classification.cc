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

#include <algorithm>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <string>

#include <r2i/r2i.h>

#define WIDTH_OFFSET 0
#define HEIGHT_OFFSET 1
#define DEFAULT_PRE_FORMAT 0
#define DEFAULT_PRE_SIZE 0

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [Model] "
            << "-s [Model Input Size] "
            << "-b [Backend] "
            << "-e [Preprocess module] "
            << "-o [Postprocess module] "
            << "-I [Input Node] (TensorFlow only) "
            << "-O [Output Node] (TensorFlow only) \n"
            << " Example: "
            << " ./classification -i cat.jpg -m graph_inceptionv2_tensorflow.pb "
            << "-s 224 -b tensorflow -e libnormalize_inceptionv1.so -o libtop_sort_postprocessing.so -I input -O Softmax"
            << std::endl;
}

std::unique_ptr<unsigned char[]> LoadImage(const std::string &path,
    int reqwidth, int reqheight, int channels) {

  int components_per_pixel;
  int width;
  int height;
  const int scaled_size = channels * reqwidth * reqheight;
  std::unique_ptr<unsigned char[]> scaled_ptr (new unsigned char[scaled_size]);
  unsigned char *scaled = nullptr;

  unsigned char *img = stbi_load(path.c_str(), &width, &height,
                                 &components_per_pixel, channels);
  if (!img) {
    std::cerr << "The picture " << path << " could not be loaded" << std::endl;
    return nullptr;
  }

  /* Scale image */
  scaled = scaled_ptr.get();
  stbir_resize_uint8(img, width, height, 0, scaled, reqwidth, reqheight, 0,
                     channels);

  std::cout << "  Image size: " << width << "x" << height << std::endl;
  std::cout << "  Scaling to: " << reqwidth << "x" << reqheight << std::endl;
  free (img);

  return scaled_ptr;
}

bool ParseArgs (int &argc, char *argv[], std::string &image_path,
                std::string &model_path, std::string &backend, int &size,
                std::string &preprocess_path, std::string &postprocess_path,
                std::string &in_node, std::string &out_node) {

  int option = 0;
  while ((option = getopt(argc, argv, "i:m:b:e:o:s:I:O:")) != -1) {
    switch (option) {
      case 'i' :
        image_path = optarg;
        break;
      case 'm' :
        model_path  = optarg;
        break;
      case 's' :
        size = std::stoi (optarg);
        break;
      case 'b' :
        backend = optarg;
        break;
      case 'e' :
        preprocess_path = optarg;
        break;
      case 'o' :
        postprocess_path = optarg;
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

std::string ToLowerCase(std::string string) {
  std::string lowercase = string;

  std::for_each(lowercase.begin(), lowercase.end(), [](char &c) {
    c = std::tolower(c);
  });
  return lowercase;
}

int main (int argc, char *argv[]) {

  r2i::RuntimeError error;
  std::string model_path;
  std::string image_path;
  std::string backend;
  std::string preprocess_path;
  std::string postprocess_path;
  std::string out_node;
  std::string in_node;
  r2i::FrameworkCode backend_code = r2i::MAX_FRAMEWORK;
  int size = 0;

  if (false == ParseArgs (argc, argv, image_path, model_path, backend,
                          size, preprocess_path, postprocess_path,
                          in_node, out_node)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty () || preprocess_path.empty()
      || postprocess_path.empty()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto backends_available = r2i::IFrameworkFactory::List (error);
  for (auto &meta : backends_available) {
    if (ToLowerCase(meta.name) == ToLowerCase(backend)) {
      backend_code = meta.code;
      break;
    }
  }

  if (backend_code == r2i::MAX_FRAMEWORK) {
    std::cerr << backend << " backend is not available"  << std::endl;
    std::cout << "Available backends are: " << std::endl;
    for (auto &meta : backends_available) {
      std::cout << "  " << meta.name << std::endl;
    }
    exit(EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   backend_code,
                   error);

  if (nullptr == factory) {
    std::cerr << backend << " backend is not built: " << error << std::endl;
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
        preprocess_path, error);
  if (error.IsError ()) {
    std::cerr << "Preprocessing loading error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  /* Get supported formats and sizes in preprocessing module */
  auto pre_formats = preprocessing->GetAvailableFormats();
  auto pre_sizes = preprocessing->GetAvailableDataSizes();

  std::cout << "Preprocessing formats supported: " << std::endl;
  for (uint i = 0 ; i < pre_formats.size(); i++) {
    auto format = pre_formats.at(i);
    std::cout << "  Format " << i << std::endl;
    std::cout << "    ID: " << format.GetId() << std::endl;
    std::cout << "    Planes: " << format.GetNumPlanes() << std::endl;
    std::cout << "    Description: " << format.GetDescription() << std::endl;
  }

  std::cout << "Preprocessing sizes supported: " << std::endl;
  for (uint i = 0 ; i < pre_sizes.size(); i++) {
    auto supported_size = pre_sizes.at(i);
    std::cout << "  Size " << i << std::endl;
    std::cout << "    Width: " << std::get<WIDTH_OFFSET>(supported_size) <<
              std::endl;
    std::cout << "    Height: " << std::get<HEIGHT_OFFSET>(supported_size) <<
              std::endl;
  }

  std::cout << "Using format " << DEFAULT_PRE_FORMAT << " and size " <<
            DEFAULT_PRE_SIZE << std::endl;
  auto pre_format = pre_formats.at(DEFAULT_PRE_FORMAT);
  auto pre_size = pre_sizes.at(DEFAULT_PRE_SIZE);
  auto pre_width = std::get<WIDTH_OFFSET>(pre_size);
  auto pre_height = std::get<HEIGHT_OFFSET>(pre_size);
  auto pre_channels = pre_format.GetNumPlanes();
  auto pre_format_id = pre_format.GetId();

  std::cout << "Setting model to engine" << std::endl;
  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine (error);
  error = engine->SetModel (model);
  if (error.IsError ()) {
    std::cerr << "Set model error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  if (backend_code == r2i::TENSORFLOW) {
    std::cout << "Configuring input and output layers" << std::endl;
    auto params = factory->MakeParameters (error);
    error = params->Configure(engine, model);
    if (error.IsError ()) {
      std::cerr << "Params configuration error: " << error << std::endl;
      exit(EXIT_FAILURE);
    }
    params->Set ("input-layer", in_node);
    params->Set ("output-layer", out_node);
  }

  std::cout << "Loading image: " << image_path << std::endl;
  std::unique_ptr<unsigned char[]> image_data = LoadImage (image_path, pre_width,
      pre_height, pre_channels);

  std::cout << "Configuring input frame" << std::endl;
  std::shared_ptr<r2i::IFrame> in_frame = factory->MakeFrame (error);

  error = in_frame->Configure (image_data.get(), pre_width, pre_height,
                               pre_format_id);
  if (error.IsError ()) {
    std::cerr << "Output frame configuration error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Configuring output frame" << std::endl;
  std::shared_ptr<r2i::IFrame> out_frame = factory->MakeFrame (error);
  std::shared_ptr<float> out_data = std::shared_ptr<float>
                                    (new float[pre_width * pre_height * pre_channels],
                                     std::default_delete<float[]>());

  error = out_frame->Configure (out_data.get(), pre_width, pre_height,
                                pre_format_id);
  if (error.IsError ()) {
    std::cerr << "Input frame configuration error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Applying pre processing" << std::endl;
  error = preprocessing->Apply (in_frame, out_frame);
  if (error.IsError ()) {
    std::cerr << "Preprocessing error: " << error.GetDescription() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Starting engine" << std::endl;
  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Engine start error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Predicting..." << std::endl;
  std::vector< std::shared_ptr<r2i::IPrediction> > predictions;
  error = engine->Predict (out_frame, predictions);

  std::cout << "Apply post processing" << std::endl;
  std::vector< std::shared_ptr<r2i::InferenceOutput> > ioutputs;
  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    loader->LoadPostprocessing(
      postprocess_path, error);
  if (error.IsError ()) {
    std::cerr << "Postprocessing loading error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  error = postprocessing->Apply(predictions, ioutputs);
  if (error.IsError ()) {
    std::cerr << "Postprocessing error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  for (uint i = 0 ; i < ioutputs.size(); i++) {
    auto ioutput = ioutputs.at(i);
    std::cout << "  Processing output: " << i << std::endl;

    if (ioutput->GetType() != r2i::CLASSIFICATION) {
      std::cerr <<
                "Wrong Inference Output type, this output is not classification type" <<
                std::endl;
      continue;
    }

    auto classification =
      std::dynamic_pointer_cast<r2i::Classification, r2i::InferenceOutput>(ioutput);
    auto labels = classification->GetLabels();

    /* Get first output only */
    std::cout << "    Highest probability is label "
              << std::get<0>(labels.at(0)) << " (" << std::get<1>(labels.at(
                    0)) << ")" << std::endl;
  }

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
