# Define common path for headers in examples directory
common_inc_dir = include_directories('common')

# Compile examples
app_examples = [
  'classification',
  'list_parameters',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach

if cdata.get('HAVE_CORAL') == true
  subdir('coral')
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

if cdata.get('HAVE_NNAPI') == true
  subdir('nnapi')
endif
