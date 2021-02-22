# Define source code
coral_sources = [
  'engine.cc',
  'frameworkfactory.cc',
]

coral_headers = [
  'engine.h',
  'frameworkfactory.h',
]

# Build library
coral_lib = static_library('coral',
  coral_sources,
  include_directories : [configinc],
  dependencies : [lib_coral_dep],
)

# Install library header files
install_headers(coral_headers, subdir : inc_install_dir + '/r2i/coral')

# Define the library as an internal dependency to the current build
internal_coral_dep = declare_dependency(link_with: coral_lib, dependencies: lib_coral_dep)
