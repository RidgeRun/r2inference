lib_r2inference_tests = [
  ['iframeworkfactory.cc'],
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

if get_option('enable-tensorflow')
  subdir('tensorflow')
endif

if get_option('enable-tflite')
  subdir('tflite')
endif
