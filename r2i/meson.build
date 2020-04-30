r2inference_internal_dep = []

if get_option('enable-tensorflow')
  subdir('tensorflow')
  r2inference_internal_dep = [internal_tf_dep]
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
install_headers(r2inference_headers, subdir : 'r2i')

# Generate pfgconfig file
pkgconfig.generate(libraries : r2inference_lib,
                  subdirs : '.',
                  description : 'C/C++ generic inference library for multiple backend frameworks',
                  name: 'r2inference-0.0')

# Define the library as an internal dependency to the current build
r2inference_lib_dep = declare_dependency(link_with: r2inference_lib, dependencies: r2inference_internal_dep)
