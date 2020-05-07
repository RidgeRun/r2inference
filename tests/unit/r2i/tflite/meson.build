subdir('resources')

lib_tflite_tests = [
  ['engine.cc'],
  ['frame.cc'],
  ['loader.cc'],
  ['parameters.cc'],
  ['prediction.cc'],
]

# Build and run tests
foreach t : lib_tflite_tests
  fname = t[0]
  test_name = fname.split('.')[0].underscorify()

  exe = executable(test_name, fname,
      include_directories : [configinc, r2i_inc_dir],
      dependencies : [r2inference_lib_dep, tests_dep, lib_tflite_dep],
  )

  # Run tests
  test(test_name, exe,
       args : '-p',
       timeout : 60,
       workdir : meson.current_build_dir())

endforeach
