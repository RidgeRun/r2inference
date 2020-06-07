onnx_tests_resources = [
  ['squeezenet.onnx'],
  ['invalid.onnx'],
]

# Copy resources files
foreach t : onnx_tests_resources
  configure_file(copy : true, input : t[0], output : t[0])
endforeach
