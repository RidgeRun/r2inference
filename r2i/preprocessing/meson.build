#Build preprocessing modules
preprocessing_sources = [
    'mean_std_preprocessing'
]

foreach pre_module : preprocessing_sources
  lib = library(pre_module, '@0@.cc'.format(pre_module),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep],
    install: false)
  internal_dep = declare_dependency(link_with: lib)
  internal_preprocessing_dep += internal_dep
endforeach
