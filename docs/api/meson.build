if get_option('enable-docs').enabled() and doxygen.found()

 #Configure env variables used in Doxyfile.in
 cdata = configuration_data()
 cdata.set('TOP_SRCDIR', meson.source_root())
 cdata.set('TOP_BUILDDIR', meson.build_root())

 doxyfile = configure_file(input: 'Doxyfile.in',
                          output: 'Doxyfile',
                          configuration: cdata,
                          install: false)

 install_docs_dir = join_paths(get_option('datadir'), 'doc', 'r2inference-@0@.0'.format(version_major))

 html_target = custom_target('docs',
                            input: doxyfile,
                            output: 'html',
                            command: [doxygen, doxyfile],
                            install: true,
                            install_dir: install_docs_dir)
endif
