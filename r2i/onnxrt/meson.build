# Define source code
onnxrt_sources = [
  'frame.cc',
  'frameworkfactory.cc',
  'loader.cc',
  'model.cc',
]

onnxrt_headers = [
  'frame.h',
  'frameworkfactory.h',
  'loader.h',
  'model.h',
]

# Build library
onnxrt_lib = static_library('onnxrt',
  onnxrt_sources,
  cpp_args : cpp_args,
  include_directories : [configinc],
  dependencies : [lib_onnxrt_dep],
)

# Install library header files
install_headers(onnxrt_headers, subdir : inc_install_dir + '/r2i/onnxrt')

# Define the library as an internal dependency to the current build
internal_onnxrt_dep = declare_dependency(link_with: onnxrt_lib, dependencies: lib_onnxrt_dep)
