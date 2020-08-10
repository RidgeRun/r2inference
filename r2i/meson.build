r2inference_internal_dep = []

if get_option('enable-tensorflow')
  subdir('tensorflow')
  r2inference_internal_dep = [internal_tf_dep]
endif

if get_option('enable-tflite')
  subdir('tflite')
  r2inference_internal_dep += [internal_tflite_dep]
endif

if get_option('enable-edgetpu')
  subdir('edgetpu')
  r2inference_internal_dep += [internal_edgetpu_dep]
endif

if get_option('enable-tensorrt')
  subdir('tensorrt')
  r2inference_internal_dep += [internal_trt_dep]
endif

if get_option('enable-onnxrt')
  subdir('onnxrt')
  r2inference_internal_dep += [internal_onnxrt_dep]
endif

if get_option('enable-onnxrt-acl')
  subdir('onnxrt_acl')
  r2inference_internal_dep += [internal_onnxrt_acl_dep]
endif

if get_option('enable-onnxrt-openvino')
  subdir('onnxrt_openvino')
  r2inference_internal_dep += [internal_onnxrt_openvino_dep]
endif

# Define source code
r2inference_sources = [
  'datatype.cc',
  'iframeworkfactory.cc',
  'imageformat.cc',
  'runtimeerror.cc',
]

r2inference_headers = [
  'datatype.h',
  'frameworkmeta.h',
  'frameworks.h',
  'iengine.h',
  'iframe.h',
  'iframeworkfactory.h',
  'iloader.h',
  'imageformat.h',
  'imodel.h',
  'iparameters.h',
  'iprediction.h',
  'parametermeta.h',
  'r2i.h',
  'runtimeerror.h',
]

# Build library
r2inference_lib = library('r2inference-@0@.0'.format(version_major),
  r2inference_sources,
  cpp_args : cpp_args,
  version : project_version,
  include_directories : [configinc],
  install : true,
  install_dir : lib_install_dir,
  dependencies : r2inference_internal_dep,
)

r2inference_static_lib = static_library('r2inference-@0@.0'.format(version_major),
  r2inference_sources,
  cpp_args : cpp_args,
  include_directories : [configinc],
  install : true,
  install_dir : lib_install_dir,
  dependencies : r2inference_internal_dep,
)

# Install library header files
install_headers(r2inference_headers, subdir : inc_install_dir + '/r2i')

# Generate pfgconfig file
pkgconfig.generate(libraries : r2inference_lib,
                  subdirs : inc_install_dir,
                  description : project_description,
                  name: 'r2inference-@0@.0'.format(version_major),
                  url: project_url)

# Define the library as an internal dependency to the current build
r2inference_lib_dep = declare_dependency(link_with: r2inference_lib, dependencies: r2inference_internal_dep)
