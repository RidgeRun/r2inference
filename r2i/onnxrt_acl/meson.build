# Define source code
onnxrt_acl_sources = [
  'engine.cc',
  'frameworkfactory.cc',
]

onnxrt_acl_headers = [
  'engine.h',
  'frameworkfactory.h',
]

# Build library
onnxrt_acl_lib = static_library('onnxrt_acl',
  onnxrt_acl_sources,
  cpp_args : cpp_args,
  include_directories : [configinc],
  dependencies : [lib_onnxrt_acl_dep],
)

# Install library header files
install_headers(onnxrt_acl_headers, subdir : inc_install_dir + '/r2i/onnxrt_acl')

# Define the library as an internal dependency to the current build
internal_onnxrt_acl_dep = declare_dependency(link_with: onnxrt_acl_lib, dependencies: lib_onnxrt_acl_dep)
