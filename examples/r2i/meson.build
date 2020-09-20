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
