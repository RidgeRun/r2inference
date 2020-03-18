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

#define NETWORK_WIDTH 416
#define NETWORK_HEIGHT 416
#define GRID_H 13
#define GRID_W 13
#define BOXES_SIZE 5
#define BOX_DIM 5
#define CLASSES 20
#define OBJ_THRESHOLD (-2.5)
#define CLASS_THRESHOLD 5

typedef struct _BBox BBox;
struct _BBox {
  int label;
  double prob;
  double x;
  double y;
  double width;
  double height;
};

static double
sigmoid (double x) {
  return 1.0 / (1.0 + pow (M_E, -1.0 * x));
}

static void
box_to_pixels (BBox *normalized_box, int row, int col, int box) {
  int grid_size = 32;
  const float box_anchors[] =
  { 1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52 };

  /* adjust the box center according to its cell and grid dim */
  normalized_box->x = (col + sigmoid (normalized_box->x)) * grid_size;
  normalized_box->y = (row + sigmoid (normalized_box->y)) * grid_size;

  /* adjust the lengths and widths */
  normalized_box->width =
    pow (M_E, normalized_box->width) * box_anchors[2 * box] * grid_size;
  normalized_box->height =
    pow (M_E, normalized_box->height) * box_anchors[2 * box + 1] * grid_size;
}

static r2i::RuntimeError
parse_boxes(std::shared_ptr<r2i::IPrediction> prediction) {
  r2i::RuntimeError error;
  float *network_output = static_cast<float *>(prediction->GetResultData());
  int i, j, b, c;

  for (i = 0; i < GRID_H; i++) {
    for (j = 0; j < GRID_W; j++) {
      for (b = 0; b < BOXES_SIZE; b++) {
        int index;
        double obj_prob;
        double cur_class_prob, max_class_prob;
        int max_class_prob_index;

        index = ((i * GRID_W + j) * BOXES_SIZE + b) * (BOX_DIM + CLASSES);
        obj_prob = network_output[index + 4];


        if (obj_prob > OBJ_THRESHOLD) {

          max_class_prob = 0;
          max_class_prob_index = 0;
          for (c = 0; c < CLASSES; c++) {
            cur_class_prob = network_output[index + BOX_DIM + c];
            if (cur_class_prob > max_class_prob) {
              max_class_prob = cur_class_prob;
              max_class_prob_index = c;
            }
          }
          if (max_class_prob > CLASS_THRESHOLD) {
            BBox result;
            result.label = max_class_prob_index;
            result.prob = max_class_prob;
            result.x = network_output[index];
            result.y = network_output[index + 1];
            result.width = network_output[index + 2];
            result.height = network_output[index + 3];
            box_to_pixels (&result, i, j, b);
            result.x = result.x - result.width * 0.5;
            result.y = result.y - result.height * 0.5;


            printf("prob: %.2f\t label: %d\n", max_class_prob, result.label);
            printf("x: %d\ty: %d\t width: %d\t height: %d\n", (int)result.x,
                   (int)result.y, (int)result.width, (int)result.height);
          }
        }
      }
    }
  }
  return error;
}

void PrintUsage() {
  std::cerr << "Required arguments: "
            << "-i [JPG input_image] "
            << "-m [YoloV2 Tensorrt Model] "
            << " Example: "
            << " ./inception -i cat.jpg -m engine.trt "
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
    adjusted[i + 0] = (static_cast<float>(scaled[i + 0])) / 255.0;
    adjusted[i + 1] = (static_cast<float>(scaled[i + 1])) / 255.0;
    adjusted[i + 2] = (static_cast<float>(scaled[i + 2])) / 255.0;
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
                std::string &model_path, int &index) {

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

  if (false == ParseArgs (argc, argv, image_path, model_path, Index)) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  if (image_path.empty() || model_path.empty ()) {
    PrintUsage ();
    exit (EXIT_FAILURE);
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   r2i::FrameworkCode::TENSORRT,
                   error);

  if (nullptr == factory) {
    std::cerr << "Tensorrt backend is not built: " << error << std::endl;
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
  std::unique_ptr<float[]> image_data = LoadImage (image_path, NETWORK_WIDTH,
                                        NETWORK_HEIGHT);

  std::cout << "Configuring frame" << std::endl;
  std::shared_ptr<r2i::IFrame> frame = factory->MakeFrame (error);

  error = frame->Configure (image_data.get(), NETWORK_WIDTH, NETWORK_HEIGHT,
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

  parse_boxes(prediction);

  std::cout << "Stopping engine" << std::endl;
  error = engine->Stop ();
  if (error.IsError ()) {
    std::cerr << "Engine stop error: " << error << std::endl;
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

