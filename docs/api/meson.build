'@0@/r2i/'.format(meson.source_root())

if get_option('enable-docs').enabled() and doxygen.found()
 conf_data = configuration_data()
 conf_data.set('DOXYGEN_INPUT', meson.source_root())
 conf_data.set('DOXYGEN_OUTPUT', meson.current_build_dir())

 doxyfile = configure_file(
   input: '@0@/common/'.format(meson.source_root()) / 'Doxyfile',
   output: 'Doxyfile',
   configuration: conf_data
 )

 custom_target('doc',
   build_by_default: true,
   command: [doxygen, doxyfile],
   output: ['html']
 )
endif
