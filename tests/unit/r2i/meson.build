lib_r2inference_tests = [
  ['iframeworkfactory.cc'],
  ['loader.cc'],
]

# Build and run tests
foreach t : lib_r2inference_tests
  fname = t[0]
  test_name = fname.split('.')[0].underscorify()

  exe = executable(test_name, fname,
      include_directories : [configinc],
      dependencies : [r2inference_lib_dep, tests_dep],
  )
  # Run tests
  test(test_name, exe, timeout : 60)

endforeach

if cdata.get('HAVE_ONNXRT') == true
  subdir('onnxrt')
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
