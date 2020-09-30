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
#include <vector>
#include <r2i/r2i.h>
#include <r2i/tensorflow/parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

void PrintTopPrediction (std::vector<std::shared_ptr<r2i::IPrediction>>
                         predictions, const std::string &path) {
  r2i::RuntimeError error;
  uint num_detections = predictions[0]->At(0, error);

  const int channels = 3;
  int width, height, cp;
  unsigned char *img = stbi_load(path.c_str(), &width, &height, &cp, channels);
  if (!img) {
    std::cerr << "The picture " << path << " could not be loaded";
    return;
  }
  cv::Mat img_mat(height, width, CV_8UC3, img);

  std::cout << "Num of detections: " << num_detections << std::endl;

  for (size_t index = 0; index < num_detections; index++) {
    std::cout << "==============================" << std::endl;
    std::cout << "Label: " << predictions[2]->At(index,
              error) << " (" << predictions[3]->At(index, error) << ")" << std::endl;

    size_t bbox_index = index * 4;
    double b_x =  predictions[1]->At(bbox_index, error);
    double b_y = predictions[1]->At(bbox_index + 1, error);
    double b_width = predictions[1]->At(bbox_index + 2, error);
    double b_height = predictions[1]->At(bbox_index + 3, error);

    std::cout << "BBox: {x:" << b_x << ", y:" << b_y << ", width:" << b_width <<
              ", height:" << b_height << "}" << std::endl;
    std::cout << "==============================" << std::endl;

    cv::Point p1(b_x * width, b_y * height);
    cv::Point p2(b_x * width + width, b_y * height + height);
    cv::rectangle(img_mat, p1, p2, cv::Scalar(0, 0, 255), 5);
  }

  cv::imshow("Prediction", img_mat);
  cv::waitKey(0);

  free(img);
}

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [Inception TensorFlow Model] "
            << "-s [Model Input Size] \n"
            << " Example: "
            << " ./ssd-mobilenet -i cat.jpg -m frozen_inference_graph.pb "
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
    adjusted[i + 0] = (static_cast<float>(scaled[i + 0]) - 128) / 128.0;
    adjusted[i + 1] = (static_cast<float>(scaled[i + 1]) - 128) / 128.0;
    adjusted[i + 2] = (static_cast<float>(scaled[i + 2]) - 128) / 128.0;
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
                std::string &model_path, int &index, int &size) {

  int option = 0;
  while ((option = getopt(argc, argv, "i:m:p:s:")) != -1) {
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
  int size = 0;

  if (false == ParseArgs (argc, argv, image_path, model_path, Index, size)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty ()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   r2i::FrameworkCode::TENSORFLOW,
                   error);

  if (nullptr == factory) {
    std::cerr << "TensorFlow backend is not built: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Loading Model: " << model_path << std::endl;
  auto loader = factory->MakeLoader (error);
  std::shared_ptr<r2i::IModel> model = loader->Load (model_path, error);
  if (error.IsError ()) {
    std::cerr << "Loader error: " << error.GetDescription() << std::endl;
    exit(EXIT_FAILURE);
  }

  std::cout << "Setting model to engine" << std::endl;
  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine (error);
  error = engine->SetModel (model);

  std::cout << "Configuring input and output layers" << std::endl;
  auto params = factory->MakeParameters (error);
  auto tf_params = static_cast<r2i::tensorflow::Parameters *>(params.get());

  error = tf_params->Configure(engine, model);

  tf_params->Set ("input-layer", "image_tensor");

  std::vector<std::string> out_nodes;
  out_nodes.push_back("num_detections");
  out_nodes.push_back("detection_boxes");
  out_nodes.push_back("detection_classes");
  out_nodes.push_back("detection_scores");
  tf_params->Set ("output-layers", out_nodes);

  std::cout << "Loading image: " << image_path << std::endl;
  std::unique_ptr<float[]> image_data = LoadImage (image_path, size,
                                        size);

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

  PrintTopPrediction(predictions, image_path);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
