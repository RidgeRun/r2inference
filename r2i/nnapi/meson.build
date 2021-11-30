# Define source code
nnapi_sources = [
  'engine.cc',
  'frameworkfactory.cc',
]

nnapi_headers = [
  'engine.h',
  'frameworkfactory.h',
]

# Build library
nnapi_lib = static_library('nnapi',
  nnapi_sources,
  include_directories : [configinc],
  dependencies : [lib_nnapi_dep],
)

# Install library header files
install_headers(nnapi_headers, subdir : inc_install_dir + '/r2i/nnapi')

# Define the library as an internal dependency to the current build
internal_nnapi_dep = declare_dependency(link_with: nnapi_lib, dependencies: lib_nnapi_dep)
