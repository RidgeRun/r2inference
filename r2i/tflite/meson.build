# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

# Define source code
tflite_sources = [
  'engine.cc',
  'frame.cc',
  'frameworkfactory.cc',
  'loader.cc',
  'model.cc',
  'parameters.cc',
  'prediction.cc',
]

tflite_headers = [
  'engine.h',
  'frame.h',
  'frameworkfactory.h',
  'loader.h',
  'model.h',
  'parameters.h',
  'prediction.h',
]

# Build library
tflite_lib = static_library('tflite-@0@'.format(project_version),
  tflite_sources,
  cpp_args : c_args,
  version : project_version,
  install : true,
  install_dir : lib_install_dir,
  include_directories : [configinc, r2i_inc_dir],
  dependencies : [lib_tflite_dep],
)

# Install library header files
install_headers(tflite_headers, subdir : 'r2inference-0.0' / 'r2i' / 'tensorflowlite')

# Define the library as an internal dependency to the current build
gst_internal_tflite_dep = declare_dependency(link_with: tflite_lib, dependencies: lib_tflite_dep, include_directories: r2i_inc_dir)
