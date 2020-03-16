
namespace nvinfer1 {
namespace {
class MockCudaEngine : public ICudaEngine {
 public:
  MockCudaEngine() {}

  ICudaEngine *deserializeCudaEngine(const void *blob, std::size_t size,
                                     IPluginFactory *pluginFactory) { return nullptr; }

  int getNbBindings() const noexcept { return 0;}

  int getBindingIndex(const char *name) const noexcept { return 0; }

  const char *getBindingName(int bindingIndex) const noexcept { return nullptr; }

  bool bindingIsInput(int bindingIndex) const noexcept { return true; }

  Dims getBindingDimensions(int bindingIndex) const noexcept { return Dims(); }

  DataType getBindingDataType(int bindingIndex) const noexcept { return DataType(); }

  int getMaxBatchSize() const noexcept { return 0; }

  int getNbLayers() const noexcept { return 0; }

  std::size_t getWorkspaceSize() const noexcept { return 0; }

  IHostMemory *serialize() const noexcept { return nullptr; }

  IExecutionContext *createExecutionContext() noexcept { return nullptr; }

  void destroy() noexcept {
    delete this;
  };

  TensorLocation getLocation(int bindingIndex) const noexcept { return TensorLocation(); }

  IExecutionContext *createExecutionContextWithoutDeviceMemory() noexcept { return nullptr; }

  size_t getDeviceMemorySize() const noexcept { return 0; }

  bool isRefittable() const noexcept { return true; }

  int getBindingBytesPerComponent(int bindingIndex) const noexcept { return 0; }

  int getBindingComponentsPerElement(int bindingIndex) const noexcept { return 0; }

  TensorFormat getBindingFormat(int bindingIndex) const noexcept { return TensorFormat(); }

  const char *getBindingFormatDesc(int bindingIndex) const noexcept { return "\0"; }

  int getBindingVectorizedDim(int bindingIndex) const noexcept { return 0; }

  const char *getName() const noexcept { return "\0"; }

  int getNbOptimizationProfiles() const noexcept { return 0; }

  Dims getProfileDimensions(int bindingIndex, int profileIndex,
                            OptProfileSelector select) const noexcept { return Dims(); }

  bool isShapeBinding(int bindingIndex) const noexcept { return true; }

  bool isExecutionBinding(int bindingIndex) { return true; }

  EngineCapability getEngineCapability() const noexcept { return EngineCapability(); }

  void setErrorRecorder(IErrorRecorder *recorder) noexcept { return; }

  IErrorRecorder *getErrorRecorder() const noexcept { return nullptr; }

  bool hasImplicitBatchDimension() const { return true; }

  const int32_t *getProfileShapeValues(int profileIndex, int inputIndex,
                                       OptProfileSelector select) const noexcept { return nullptr; }

  bool isExecutionBinding(int bindingIndex) const noexcept { return true; }

};
}
}
