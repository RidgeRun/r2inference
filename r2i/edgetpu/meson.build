# Define source code
edgetpu_sources = [
  'engine.cc',
  'frameworkfactory.cc',
  'parameters.cc'
]

edgetpu_headers = [
  'engine.h',
  'frameworkfactory.h',
  'parameters.h'
]

# Build library
edgetpu_lib = static_library('edgetpu',
  edgetpu_sources,
  include_directories : [configinc],
  dependencies : [lib_edgetpu_dep],
)

# Install library header files
install_headers(edgetpu_headers, subdir : inc_install_dir + '/r2i/edgetpu')

# Define the library as an internal dependency to the current build
internal_edgetpu_dep = declare_dependency(link_with: edgetpu_lib, dependencies: lib_edgetpu_dep)
