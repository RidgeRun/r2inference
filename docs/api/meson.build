# Copy the file provided in input: to a file in the build directory with
# the name output: in the current directory.
types = configure_file(input : 'r2inference.types',
  output : 'r2inference.types',
  copy : true)

# Compiles and installs gtkdoc documentation into prefix/share/gtk-doc/html.
gnome.gtkdoc('r2inference',
  main_xml : 'r2inference-docs.xml',
  src_dir : ['@0@/r2i/'.format(meson.source_root()) ],
  mkdb_args : ['--xml-mode', '--output-format=xml' ],
  html_args : '--verbose',
  gobject_typesfile : types,
  dependencies : [lib_tf_dep] + [lib_tflite_dep] + [r2inference_lib_dep],
  install : true
)
