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
tf_sources = [
  'engine.cc',
  'frame.cc',
  'frameworkfactory.cc',
  'loader.cc',
  'model.cc',
  'parameters.cc',
  'prediction.cc',
]

tf_headers = [
  'engine.h',
  'frame.h',
  'frameworkfactory.h',
  'loader.h',
  'model.h',
  'parameters.h',
  'prediction.h',
]

# Build library
tf_lib = library('tf-@0@'.format(project_version),
  tf_sources,
  c_args : c_args,
  version : project_version,
  install : true,
  install_dir : lib_install_dir,
  include_directories : [configinc, r2i_inc_dir],
  dependencies : [lib_tf_dep, r2inference_lib_dep],
)

lib_tf_dir = meson.current_source_dir()

# Define the library as an internal dependency to the current build
gst_internal_dep = declare_dependency(link_with: tf_lib)
