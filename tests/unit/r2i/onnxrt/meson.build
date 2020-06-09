subdir('resources')

lib_onnxrt_tests = [
  ['frame.cc'],
  ['frameworkfactory.cc'],
  ['loader.cc'],
  ['model.cc'],
]

# Build and run tests
foreach t : lib_onnxrt_tests
  fname = t[0]
  test_name = fname.split('.')[0].underscorify()

  exe = executable(test_name, fname,
      include_directories : [configinc],
      dependencies : [r2inference_lib_dep, tests_dep, lib_onnxrt_dep],
  )

  # Run tests
  test(test_name, exe,
       args : '-p',
       timeout : 60,
       workdir : meson.current_build_dir())

endforeach
