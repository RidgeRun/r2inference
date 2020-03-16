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

#include "r2i/tensorrt/loader.h"

#include <fstream>
#include <memory>

#include "r2i/imodel.h"
#include "r2i/tensorrt/model.h"

namespace r2i {
namespace tensorrt {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {
  std::ifstream graph_file;
  // char *buf_data = nullptr;

  if (in_path.empty()) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Received NULL path to file");
    return nullptr;
  }

  std::ifstream graphdef_file;
  graphdef_file.open (in_path, std::ios::binary | std::ios::ate);
  if (false == graphdef_file.is_open()) {
    error.Set (RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }

  graph_file.open (in_path, std::ios::binary | std::ios::ate);
  if (graph_file.is_open()) {
    std::shared_ptr<Model> model (new r2i::tensorrt::Model());
    unsigned int graph_size;

    /* Get file size */
    graph_size = graph_file.tellg();
    graph_file.seekg (0, std::ios::beg);

    std::shared_ptr<void> graph_data(malloc (graph_size), free);

    if (nullptr == graph_data.get()) {
      error.Set (RuntimeError::Code::MEMORY_ERROR,
                 "Can not allocate memory for graph");
      graph_file.close ();
      return nullptr;
    }

    /* Store the content of the graph */
    graph_file.read (reinterpret_cast<char *>(graph_data.get()), graph_size);
    if (!graph_file) {
      error.Set (RuntimeError::Code::FILE_ERROR, "Can not read file");
      graph_file.close ();
      return nullptr;
    }

    this->model = model;
    graph_file.close ();
  } else {
    error.Set (RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }

  return nullptr;
}
}
}

