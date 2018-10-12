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

void print_usage() {
  printf("Usage: example -i [input_image] -m [Model] \n");
}

int main (int argc, char *argv[]) {
  std::shared_ptr<r2i::IModel> model;
  r2i::ncsdk::Engine engine;
  r2i::ncsdk::Loader loader;
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
  model = loader.Load (model_path, error);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Loader Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  error = engine.SetModel (model);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine SetModel Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  error = engine.Start ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Start Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }

  error = engine.Stop ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    printf("Engine Stop Error: %s\n", error.GetDescription().c_str());
    exit(EXIT_FAILURE);
  }
  return 0;
}
