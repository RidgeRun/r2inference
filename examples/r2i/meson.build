# Define common headers directory for examples
common_inc_dir = include_directories('common')

# Compile examples
app_examples = [
  'list_parameters',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach

if cdata.get('HAVE_EDGETPU') == true
  subdir('edgetpu')
endif

if cdata.get('HAVE_ONNXRT') == true
  subdir('onnxrt')
endif

if cdata.get('HAVE_ONNXRT_ACL') == true
  subdir('onnxrt_acl')
endif

if cdata.get('HAVE_ONNXRT_OPENVINO') == true
  subdir('onnxrt_openvino')
endif

if cdata.get('HAVE_TENSORFLOW') == true
  subdir('tensorflow')
endif

if cdata.get('HAVE_TFLITE') == true
  subdir('tflite')
endif

if cdata.get('HAVE_TENSORRT') == true
  subdir('tensorrt')
endif

preprocessing_sources = [
    'mean_std_preprocessing'
]

foreach pre_module : preprocessing_sources
  library(pre_module, '@0@.cc'.format(pre_module),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach

postprocessing_sources = [
    'top_sort_postprocessing'
]

foreach post_module : postprocessing_sources
  library(post_module, '@0@.cc'.format(post_module),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach
