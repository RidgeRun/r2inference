# Compile examples
app_examples = [
  'inception',
]

example_dependencies = [
  r2inference_lib_dep
]

if opencv_dep.found()
  app_examples += ['tf1-object-detection-api']
  example_dependencies += [opencv_dep]
endif

foreach app : app_examples
  executable(app, '@0@.cc'.format(app),
    include_directories: [configinc, common_inc_dir],
    dependencies : example_dependencies,
    install: false)
endforeach
