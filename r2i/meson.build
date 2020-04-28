# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if get_option('enable-tensorflow')
  subdir('tensorflow')
  r2inference_internal_dep = [gst_internal_tf_dep]
endif

if get_option('enable-tflite')
  subdir('tflite')
  r2inference_internal_dep += [gst_internal_tflite_dep]
endif

# Define source code
r2inference_sources = [
  'datatype.cc',
  'iframeworkfactory.cc',
  'imageformat.cc',
  'runtimeerror.cc',
]

r2inference_headers = [
  'datatype.h',
  'frameworkmeta.h',
  'frameworks.h',
  'iengine.h',
  'iframe.h',
  'iframeworkfactory.h',
  'iloader.h',
  'imageformat.h',
  'imodel.h',
  'iparameters.h',
  'iprediction.h',
  'parametermeta.h',
  'r2i.h',
  'runtimeerror.h',
]

# Build library
r2inference_lib = library('r2inference-@0@'.format(project_version),
  r2inference_sources,
  cpp_args : c_args,
  version : project_version,
  include_directories : [configinc],
  install : true,
  install_dir : lib_install_dir,
  dependencies : r2inference_internal_dep,
)

# Install library header files
install_headers(r2inference_headers, subdir : '../r2inference-0.0' / 'r2i')

# Generate pfgconfig file
pkgconfig.generate(libraries : r2inference_lib,
                  subdirs : '.',
                  description : 'C/C++ generic inference library for multiple backend frameworks',
                  name: 'r2inference-0.0')

lib_r2inference_dir = meson.current_source_dir()

# Define the library as an internal dependency to the current build
r2inference_lib_dep = declare_dependency(link_with: r2inference_lib, dependencies: r2inference_internal_dep)
