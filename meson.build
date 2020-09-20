project('r2inference', ['cpp'], default_options : ['cpp_std=c++11'],
  version : '0.10.0',
  meson_version : '>= 0.50',)

# Set project information
project_description = 'C/C++ generic inference library for multiple backend frameworks'
project_url = 'https://github.com/RidgeRun/r2inference/'
project_name = meson.project_name()
project_version = meson.project_version()
version_arr = project_version.split('.')
version_major = version_arr[0].to_int()
version_minor = version_arr[1].to_int()
version_micro = version_arr[2].to_int()
if version_arr.length() == 4
  version_nano = version_arr[3].to_int()
else
  version_nano = 0
endif
version_is_dev = version_minor % 2 == 1 and version_micro < 90

# Define common path for headers in examples directory
common_inc_dir = include_directories('examples/r2i/common')

# Get an object returns describing a compiler
cpp = meson.get_compiler('cpp')

# Define compiler args and include directories
cpp_args = ['-DHAVE_CONFIG_H']

# Create an empty configuration object to set config.h information
cdata = configuration_data()

# define common dependencies
glib = dependency('glib-2.0', version:'>= 2.0.0', required: true)
gmodule = dependency('gmodule-2.0', version:'>= 2.0.0', required: true)
common_deps = [glib, gmodule]

# Initialize preprocessor definitions
cdata.set('HAVE_TENSORFLOW', false)
cdata.set('HAVE_TFLITE', false)
cdata.set('HAVE_EDGETPU', false)
cdata.set('HAVE_TENSORRT', false)
cdata.set('HAVE_ONNXRT', false)
cdata.set('HAVE_ONNXRT_ACL', false)
cdata.set('HAVE_ONNXRT_OPENVINO', false)

# Define library dependencies for Tensorflow support
if get_option('enable-tensorflow')
  tensorflow = cpp.find_library('tensorflow', required: true)
  tensorflow_dep = declare_dependency(dependencies: tensorflow)
  lib_tf_dep = [tensorflow_dep, common_deps]
  cdata.set('HAVE_TENSORFLOW', true)
endif

# Define library dependencies for Tensorflow Lite support
if get_option('enable-tflite') or get_option('enable-edgetpu')
  tensorflow_lite = cpp.find_library('tensorflow-lite', required: true)
  tensorflow_lite_dep = declare_dependency(dependencies: tensorflow_lite)
  thread_dep = dependency('threads')
  lib_tflite_dep = [tensorflow_lite_dep, thread_dep, common_deps]
  cdata.set('HAVE_TFLITE', true)
endif

# Define library dependencies for Tensorflow Lite support
if get_option('enable-edgetpu')
    edgetpu = cpp.find_library(':libedgetpu.so.1.0', required: true)
    edgetpu_dep = declare_dependency(dependencies: edgetpu)
    thread_dep = dependency('threads')
    dl_dep = cpp.find_library('dl', required : true)
    rt_dep = cpp.find_library('rt', required : true)
    lib_edgetpu_dep = [edgetpu_dep, lib_tflite_dep, dl_dep, rt_dep, common_deps]
    cdata.set('HAVE_EDGETPU', true)
endif

# Define library dependencies for TensorRT support
if get_option('enable-tensorrt')

  # CUDA versions supported
  cuda_versions = [
    '10.0',
  ]

  # Find CUDA version
  cuda_ver = ''
  foreach v : cuda_versions
    cuda_dep = dependency('cudart-' + v, required: false)
    cudart_dep = dependency('cudart-' + v, required: false)
    if cuda_dep.found() and cudart_dep.found()
      cuda_ver = v
      break
    endif
  endforeach

  if not cuda_dep.found() or not cudart_dep.found()
    error ('Unable to find CUDA supported version')
  endif

  tensorrt = cpp.find_library('nvinfer', required: true)
  tensorrt_dep = declare_dependency(dependencies: tensorrt)
  lib_trt_dep = [tensorrt_dep, cuda_dep, cudart_dep, common_deps]
  cdata.set('HAVE_TENSORRT', true)
endif

# Define library dependencies for ONNX support
if get_option('enable-onnxrt') or get_option('enable-onnxrt-acl') or get_option('enable-onnxrt-openvino')
  onnxrt = cpp.find_library('onnxruntime', required: true)
  onnxrt_dep = declare_dependency(dependencies: onnxrt)
  lib_onnxrt_dep = [onnxrt_dep, common_deps]
  cdata.set('HAVE_ONNXRT', true)
endif

# Define library dependencies for ONNXRT with ACL execution provider support
if get_option('enable-onnxrt-acl')
  arm_compute = cpp.find_library('arm_compute', required: true)
  arm_compute_dep = declare_dependency(dependencies: arm_compute)
  arm_compute_core = cpp.find_library('arm_compute_core', required: true)
  arm_compute_core_dep = declare_dependency(dependencies: arm_compute_core)
  arm_compute_graph = cpp.find_library('arm_compute_graph', required: true)
  arm_compute_graph_dep = declare_dependency(dependencies: arm_compute_graph)
  lib_onnxrt_acl_dep = [arm_compute_dep, arm_compute_core_dep, arm_compute_graph_dep, lib_onnxrt_dep, common_deps]
  lib_onnxrt_dep += [lib_onnxrt_acl_dep]
  cdata.set('HAVE_ONNXRT_ACL', true)
endif

# Define library dependencies for ONNXRT with OpenVINO execution provider support
if get_option('enable-onnxrt-openvino')
  cdata.set('HAVE_ONNXRT_OPENVINO', true)
endif

# Check if at least one backend is enabled
if (cdata.get('HAVE_TENSORFLOW') == false and cdata.get('HAVE_TFLITE') == false and
        cdata.get('HAVE_TENSORRT') == false and cdata.get('HAVE_ONNXRT') == false and
        cdata.get('HAVE_ONNXRT_ACL') == false and cdata.get('HAVE_ONNXRT_OPENVINO') == false)
  error ('No backend selected, you must choose at least one')
endif

# Find dependencies for TESTS
if get_option('enable-tests').enabled()
  tests_dep = dependency('cpputest', required: true)
endif

# Doxygen documentation
if get_option('enable-docs').enabled()
  doxygen = find_program('doxygen', required: true)
endif

# Define header directories
configinc = include_directories('.')

# Define installation directories
prefix = get_option('prefix')
lib_install_dir = get_option('libdir')
inc_install_dir = 'r2inference-@0@.0'.format(version_major)

# Set config.h information
cdata.set_quoted('LOCALEDIR', join_paths(get_option('prefix'), get_option('localedir')))
cdata.set_quoted('LIBDIR', join_paths(get_option('prefix'), get_option('libdir')))
cdata.set_quoted('PACKAGE', project_name)
cdata.set_quoted('PACKAGE_NAME', project_name)
cdata.set_quoted('PACKAGE_STRING', '@0@ @1@'.format(project_name,project_version))
cdata.set_quoted('PACKAGE_TARNAME', project_name)
cdata.set_quoted('PACKAGE_BUGREPORT', ' https://github.com/RidgeRun/r2inference/issues')
cdata.set_quoted('PACKAGE_VERSION', project_version)
cdata.set_quoted('VERSION', project_version)

# These are only needed/used by the ABI tests
host_defines = [
  [ 'x86', 'HAVE_CPU_I386' ],
  [ 'x86_64', 'HAVE_CPU_X86_64' ],
  [ 'arm', 'HAVE_CPU_ARM' ],
  [ 'aarch64', 'HAVE_CPU_AARCH64' ],
  [ 'mips', 'HAVE_CPU_MIPS' ],
  [ 'powerpc', 'HAVE_CPU_PPC' ],
  [ 'powerpc64', 'HAVE_CPU_PPC64' ],
  [ 'alpha', 'HAVE_CPU_ALPHA' ],
  [ 'sparc', 'HAVE_CPU_SPARC' ],
  [ 'ia64', 'HAVE_CPU_IA64' ],
  [ 'hppa', 'HAVE_CPU_HPPA' ],
  [ 'm68k', 'HAVE_CPU_M68K' ],
  [ 's390', 'HAVE_CPU_S390' ],
]
foreach h : host_defines
  if h.get(0) == host_machine.cpu()
    cdata.set(h.get(1), 1)
  else
    cdata.set(h.get(1), false)
  endif
endforeach

cdata.set_quoted('HOST_CPU', host_machine.cpu())

# Imports pkgconfig module
pkgconfig = import('pkgconfig')

# Install git hooks
python3 = import('python').find_installation()
run_command(python3, '-c', 'import shutil; shutil.copy("hooks/pre-commit.hook", ".git/hooks/pre-commit")')

# Meson will generate a header file all the entries in the configuration data object
configure_file(output : 'config.h', configuration : cdata)

# Enter to each subdirectory and execute the meson.build
subdir('r2i')
subdir('tests')
subdir('examples')
subdir('docs')
