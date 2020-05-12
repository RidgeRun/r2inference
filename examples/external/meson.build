# Compile examples
app_examples = [
  'list_backends',
]

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc],
    dependencies : [r2inference_lib_dep],
    install: false)
endforeach
