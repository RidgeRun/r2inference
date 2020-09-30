# Compile examples
app_examples = [
  'inception',
  'ssd-mobilenet',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc, common_inc_dir],
    dependencies : [r2inference_lib_dep, opencv_dep],
    install: false)
endforeach
