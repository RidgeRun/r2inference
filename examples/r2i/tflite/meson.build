# Compile examples
app_examples = [
  'inception',
]

common_inc_dir = '../common/'

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc, r2i_inc_dir, common_inc_dir],
    dependencies : [r2inference_lib_dep, gst_internal_tflite_dep],
    install: false)
endforeach
