

namespace nvinfer1 {
namespace {
class MockCudaEngine : public ICudaEngine {
 public:
  MockCudaEngine() {}

  MockCudaEngine *deserializeCudaEngine(const void *blob, std::size_t size,
                                        IPluginFactory *pluginFactory) { return nullptr; }

  int getNbBindings() const throw() { return 0;}

  int getBindingIndex(const char *name) const throw() { return 0; }

  const char *getBindingName(int bindingIndex) const throw() { return nullptr; }

  bool bindingIsInput(int bindingIndex) const throw() { return true; }

  Dims getBindingDimensions(int bindingIndex) const throw() { return Dims(); }

  DataType getBindingDataType(int bindingIndex) const throw() { return DataType(); }

  int getMaxBatchSize() const throw() { return 0; }

  int getNbLayers() const throw() { return 0; }

  std::size_t getWorkspaceSize() const throw() { return 0; }

  IHostMemory *serialize() const throw() { return nullptr; }

  IExecutionContext *createExecutionContext() throw() { return nullptr; }

  void destroy() noexcept {
    delete this;
  };

  TensorLocation getLocation(int bindingIndex) const throw() { return TensorLocation(); }

  IExecutionContext *createExecutionContextWithoutDeviceMemory() throw() { return nullptr; }

  size_t getDeviceMemorySize() const throw() { return 0; }

  bool isRefittable() const throw() { return true; }

  int getBindingBytesPerComponent(int bindingIndex) const throw() { return 0; }

  int getBindingComponentsPerElement(int bindingIndex) const throw() { return 0; }

  TensorFormat getBindingFormat(int bindingIndex) const throw() { return TensorFormat(); }

  const char *getBindingFormatDesc(int bindingIndex) const throw() { return "\0"; }

  int getBindingVectorizedDim(int bindingIndex) const throw() { return 0; }

  const char *getName() const throw() { return "\0"; }

  int getNbOptimizationProfiles() const throw() { return 0; }

  Dims getProfileDimensions(int bindingIndex, int profileIndex,
                            OptProfileSelector select) const throw() { return Dims(); }

  bool isShapeBinding(int bindingIndex) const throw() { return true; }

  bool isExecutionBinding(int bindingIndex) { return true; }

  EngineCapability getEngineCapability() const throw() { return EngineCapability(); }

  void setErrorRecorder(IErrorRecorder *recorder) noexcept { return; }

  IErrorRecorder *getErrorRecorder() const throw() { return nullptr; }

  bool hasImplicitBatchDimension() const { return true; }

  const int32_t *getProfileShapeValues(int profileIndex, int inputIndex,
                                       OptProfileSelector select) const throw() { return nullptr; }

  bool isExecutionBinding(int bindingIndex) const throw() { return true; }

};
}
}
