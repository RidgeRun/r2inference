# Define common headers directory for examples
common_inc_dir = include_directories('common')

if get_option('enable-tensorflow')
  subdir('tensorflow')
endif

if get_option('enable-tflite')
  subdir('tflite')
endif

if get_option('enable-tensorrt')
  subdir('tensorrt')
endif
