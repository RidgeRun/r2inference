# Compile examples
app_examples = [
  'inception',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc, r2i_inc_dir, common_inc_dir],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach
