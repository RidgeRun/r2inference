# Define source code
trt_sources = [
  'engine.cc',
  'frame.cc',
  'frameworkfactory.cc',
  'loader.cc',
  'model.cc',
  'parameters.cc',
  'prediction.cc',
]

trt_headers = [
  'engine.h',
  'frame.h',
  'frameworkfactory.h',
  'loader.h',
  'model.h',
  'parameters.h',
  'prediction.h',
]

# Build library
trt_lib = static_library('trt',
  trt_sources,
  include_directories : [configinc],
  dependencies : [lib_trt_dep],
)

# Install library header files
install_headers(trt_headers, subdir : inc_install_dir + '/r2i/tensorrt')

# Define the library as an internal dependency to the current build
internal_trt_dep = declare_dependency(link_with: trt_lib, dependencies: lib_trt_dep)
