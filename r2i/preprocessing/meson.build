#Build preprocessing modules
preprocessing_sources = [
    'normalize_inceptionv1',
    'normalize_inceptionv3',
    'normalize_resnet50v1',
]

normalize_lib = static_library('normalize',
  'normalize.cc',
  cpp_args : cpp_args,
  dependencies : [r2inference_lib_dep],
  include_directories : [configinc, common_inc_dir],
)

internal_normalize_dep = declare_dependency(link_with: normalize_lib)

foreach pre_module : preprocessing_sources
  lib = library(pre_module, '@0@.cc'.format(pre_module),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep, internal_normalize_dep],
    install: false)
  internal_dep = declare_dependency(link_with: lib)
  internal_preprocessing_dep += internal_dep
endforeach
