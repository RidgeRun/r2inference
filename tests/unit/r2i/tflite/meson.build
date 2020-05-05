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
  has_flags = t.get(1,false)
  flags = t.get(2,'')

  if has_flags
    test_flags = c_args + [flags]
  else
    test_flags = c_args
  endif

  exe = executable(test_name, fname,
      c_args : test_flags,
      cpp_args : test_flags,
      include_directories : [configinc, r2i_inc_dir],
      dependencies : [r2inference_lib_dep, tests_dep, lib_tflite_dep],
  )
  # Run tests
  test(test_name, exe, timeout : 60)

endforeach
