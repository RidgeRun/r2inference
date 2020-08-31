# Define source code
tf_sources = [
  'engine.cc',
  'frame.cc',
  'frameworkfactory.cc',
  'loader.cc',
  'model.cc',
  'parameters.cc',
]

tf_headers = [
  'engine.h',
  'frame.h',
  'frameworkfactory.h',
  'loader.h',
  'model.h',
  'parameters.h',
]

# Build library
tf_lib = static_library('tf',
  tf_sources,
  include_directories : [configinc],
  dependencies : [lib_tf_dep],
)

# Install library header files
install_headers(tf_headers, subdir : inc_install_dir + '/r2i/tensorflow')

# Define the library as an internal dependency to the current build
internal_tf_dep = declare_dependency(link_with: tf_lib, dependencies: lib_tf_dep)
