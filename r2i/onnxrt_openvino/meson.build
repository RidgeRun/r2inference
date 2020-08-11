# Define source code
onnxrt_openvino_sources = [
  'engine.cc',
  'frameworkfactory.cc',
]

onnxrt_openvino_headers = [
  'engine.h',
  'frameworkfactory.h',
]

# Build library
onnxrt_openvino_lib = static_library('onnxrt_openvino',
  onnxrt_openvino_sources,
  cpp_args : cpp_args,
  include_directories : [configinc],
)

# Install library header files
install_headers(onnxrt_openvino_headers, subdir : inc_install_dir + '/r2i/onnxrt_openvino')

# Define the library as an internal dependency to the current build
internal_onnxrt_openvino_dep = declare_dependency(link_with: onnxrt_openvino_lib)
