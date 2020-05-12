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
tflite_lib = static_library('tflite',
  tflite_sources,
  include_directories : [configinc],
  dependencies : [lib_tflite_dep],
)

# Install library header files
install_headers(tflite_headers, subdir : inc_install_dir + '/r2i/tflite')

# Define the library as an internal dependency to the current build
internal_tflite_dep = declare_dependency(link_with: tflite_lib, dependencies: lib_tflite_dep)
