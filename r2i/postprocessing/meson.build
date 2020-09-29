#Build postprocessing modules
postprocessing_sources = [
    'top_sort_postprocessing'
]

foreach post_module : postprocessing_sources
  lib = library(post_module, '@0@.cc'.format(post_module),
    include_directories: [configinc],
    dependencies : [r2inference_lib_dep],
    install: false)
  internal_dep = declare_dependency(link_with: lib)
  internal_postprocessing_dep += internal_dep
endforeach
