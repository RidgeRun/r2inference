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

#include "r2i/tensorflow/loader.h"

#include <fstream>
#include <memory>
#include <tensorflow/c/c_api.h>

#include "r2i/imodel.h"
#include "r2i/tensorflow/model.h"

namespace r2i {
namespace tensorflow {

static void free_buffer (void *data, size_t size) {
  free (data);
}

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {
  char *buf_data = nullptr;

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

  /* Get file size */
  unsigned int graphdef_size = graphdef_file.tellg();
  graphdef_file.seekg (0, std::ios::beg);

  std::shared_ptr<TF_Buffer> buf(TF_NewBuffer (), TF_DeleteBuffer);
  buf_data = reinterpret_cast<char *>(malloc (graphdef_size));
  buf->data = buf_data;
  buf->data_deallocator = free_buffer;
  buf->length = graphdef_size;

  if (nullptr == buf->data) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Can not allocate memory for graphdef");
    return nullptr;
  }

  /* Read the contents of the graphdef */
  graphdef_file.read (buf_data, buf->length);
  if (!graphdef_file) {
    error.Set (RuntimeError::Code::FILE_ERROR, "Unable to read file");
    return nullptr;
  }

  auto model = std::make_shared<Model>();
  error = model->Load (buf);
  if (error.IsError ()) {
    model = nullptr;
  }

  return model;
}

}
}
