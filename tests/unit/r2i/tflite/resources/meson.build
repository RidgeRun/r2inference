tflite_tests_resources = [
  ['squeezenet.tflite'],
  ['invalid.tflite'],
]

# Copy resources files
foreach t : tflite_tests_resources
  configure_file(copy : true, input : t[0], output : t[0])
endforeach
