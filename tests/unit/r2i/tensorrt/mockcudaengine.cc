bool fail_context = false;
bool execute_error = false;

#define MAX_BATCH_SIZE 32

namespace nvinfer1 {
namespace {
class MockCudaEngine : public ICudaEngine {
 public:
  MockCudaEngine();

  ICudaEngine *deserializeCudaEngine(const void *blob, std::size_t size,
                                     IPluginFactory *pluginFactory);

  int getNbBindings() const noexcept;

  int getBindingIndex(const char *name) const noexcept;

  const char *getBindingName(int bindingIndex) const noexcept;

  bool bindingIsInput(int bindingIndex) const noexcept;

  Dims getBindingDimensions(int bindingIndex) const noexcept;

  DataType getBindingDataType(int bindingIndex) const noexcept;

  int getMaxBatchSize() const noexcept;

  int getNbLayers() const noexcept;

  std::size_t getWorkspaceSize() const noexcept;

  IHostMemory *serialize() const noexcept;

  IExecutionContext *createExecutionContext() noexcept;

  void destroy() noexcept;

  TensorLocation getLocation(int bindingIndex) const noexcept;

  IExecutionContext *createExecutionContextWithoutDeviceMemory() noexcept;

  size_t getDeviceMemorySize() const noexcept;

  bool isRefittable() const noexcept;

  int getBindingBytesPerComponent(int bindingIndex) const noexcept;

  int getBindingComponentsPerElement(int bindingIndex) const noexcept;

  TensorFormat getBindingFormat(int bindingIndex) const noexcept;

  const char *getBindingFormatDesc(int bindingIndex) const noexcept;

  int getBindingVectorizedDim(int bindingIndex) const noexcept;

  const char *getName() const noexcept;

  int getNbOptimizationProfiles() const noexcept;

  Dims getProfileDimensions(int bindingIndex, int profileIndex,
                            OptProfileSelector select) const noexcept;

  bool isShapeBinding(int bindingIndex) const noexcept;

  bool isExecutionBinding(int bindingIndex);

  EngineCapability getEngineCapability() const noexcept;

  void setErrorRecorder(IErrorRecorder *recorder) noexcept;

  IErrorRecorder *getErrorRecorder() const noexcept;

  bool hasImplicitBatchDimension() const;

  const int32_t *getProfileShapeValues(int profileIndex, int inputIndex,
                                       OptProfileSelector select) const noexcept;

  bool isExecutionBinding(int bindingIndex) const noexcept;
};


class MockExecutionContext : public IExecutionContext {
 public:
  MockCudaEngine cuda_engine;

  MockExecutionContext();

  bool execute(int batchSize, void **bindings) noexcept;

  bool enqueue(int batchSize, void **bindings, cudaStream_t stream,
               cudaEvent_t *inputConsumed) noexcept;

  void setDebugSync(bool sync) noexcept;

  bool getDebugSync() const noexcept;

  void setProfiler(IProfiler *) noexcept;

  IProfiler *getProfiler() const noexcept;

  const ICudaEngine &getEngine() const noexcept;

  void destroy() noexcept;

  void setName(const char *name) noexcept;

  const char *getName() const noexcept;

  void setDeviceMemory(void *memory) noexcept;

  Dims getStrides(int bindingIndex) const noexcept;

  bool setOptimizationProfile(int profileIndex) noexcept;

  int getOptimizationProfile() const noexcept;

  bool setBindingDimensions(int bindingIndex, Dims dimensions) noexcept;

  Dims getBindingDimensions(int bindingIndex) const noexcept;

  bool setInputShapeBinding(int bindingIndex, const int32_t *data) noexcept;

  bool getShapeBinding(int bindingIndex, int32_t *data) const noexcept;

  bool allInputDimensionsSpecified() const noexcept;

  bool allInputShapesSpecified() const noexcept;

  void setErrorRecorder(IErrorRecorder *recorder) noexcept;

  IErrorRecorder *getErrorRecorder() const noexcept;

  bool executeV2(void **bindings) noexcept;

  bool enqueueV2(void **bindings, cudaStream_t stream,
                 cudaEvent_t *inputConsumed) noexcept;

};

MockCudaEngine::MockCudaEngine() {}

int MockCudaEngine::getNbBindings() const noexcept {
  /* This values is used as a default since 1 input and 1 output  */
  return 2;
}

int MockCudaEngine::getBindingIndex(const char *name) const noexcept { return 0; }

const char *MockCudaEngine::getBindingName(int bindingIndex) const noexcept { return nullptr; }

bool MockCudaEngine::bindingIsInput(int bindingIndex) const noexcept { return true; }

Dims MockCudaEngine::getBindingDimensions(int bindingIndex) const noexcept { return Dims(); }

DataType MockCudaEngine::getBindingDataType(int bindingIndex) const noexcept { return DataType(); }

int MockCudaEngine::getMaxBatchSize() const noexcept {
  return MAX_BATCH_SIZE;
}

int MockCudaEngine::getNbLayers() const noexcept { return 0; }

std::size_t MockCudaEngine::getWorkspaceSize() const noexcept { return 0; }

IHostMemory *MockCudaEngine::serialize() const noexcept { return nullptr; }

IExecutionContext *MockCudaEngine::createExecutionContext() noexcept {
  nvinfer1::MockExecutionContext *ptr;

  if (!fail_context) {
    ptr = new MockExecutionContext();
  } else {
    ptr = nullptr;
  }

  return ptr;
}

void MockCudaEngine::destroy() noexcept {
  delete this;
};

TensorLocation MockCudaEngine::getLocation(int bindingIndex) const noexcept { return TensorLocation(); }

IExecutionContext *MockCudaEngine::createExecutionContextWithoutDeviceMemory()
noexcept { return nullptr; }

size_t MockCudaEngine::getDeviceMemorySize() const noexcept { return 0; }

bool MockCudaEngine::isRefittable() const noexcept { return true; }

int MockCudaEngine::getBindingBytesPerComponent(int bindingIndex) const
noexcept { return 0; }

int MockCudaEngine::getBindingComponentsPerElement(int bindingIndex) const
noexcept { return 0; }

TensorFormat MockCudaEngine::getBindingFormat(int bindingIndex) const noexcept { return TensorFormat(); }

const char *MockCudaEngine::getBindingFormatDesc(int bindingIndex) const
noexcept { return "\0"; }

int MockCudaEngine::getBindingVectorizedDim(int bindingIndex) const noexcept { return 0; }

const char *MockCudaEngine::getName() const noexcept { return "\0"; }

int MockCudaEngine::getNbOptimizationProfiles() const noexcept { return 0; }

Dims MockCudaEngine::getProfileDimensions(int bindingIndex, int profileIndex,
    OptProfileSelector select) const noexcept { return Dims(); }

bool MockCudaEngine::isShapeBinding(int bindingIndex) const noexcept { return true; }

//bool MockCudaEngine::isExecutionBinding(int bindingIndex) { return true; }

EngineCapability MockCudaEngine::getEngineCapability() const noexcept { return EngineCapability(); }

void MockCudaEngine::setErrorRecorder(IErrorRecorder *recorder) noexcept { return; }

IErrorRecorder *MockCudaEngine::getErrorRecorder() const noexcept { return nullptr; }

bool MockCudaEngine::hasImplicitBatchDimension() const { return true; }

const int32_t *MockCudaEngine::getProfileShapeValues(int profileIndex,
    int inputIndex,
    OptProfileSelector select) const noexcept { return nullptr; }

bool MockCudaEngine::isExecutionBinding(int bindingIndex) const noexcept { return true; }

};

MockExecutionContext::MockExecutionContext() {}

bool MockExecutionContext::execute(int batchSize, void **bindings) noexcept {
  return !execute_error;
}

bool MockExecutionContext::enqueue(int batchSize, void **bindings,
                                   cudaStream_t stream, cudaEvent_t *inputConsumed) noexcept { return true; }

void MockExecutionContext::setDebugSync(bool sync) noexcept { return; }

bool MockExecutionContext::getDebugSync() const noexcept { return true; }

void MockExecutionContext::setProfiler(IProfiler *) noexcept { return; }

IProfiler *MockExecutionContext::getProfiler() const noexcept { return nullptr; }

const ICudaEngine &MockExecutionContext::getEngine() const noexcept { return cuda_engine; }

void MockExecutionContext::destroy() noexcept { delete this; }

void MockExecutionContext::setName(const char *name) noexcept { return; }

const char *MockExecutionContext::getName() const noexcept { return nullptr; }

void MockExecutionContext::setDeviceMemory(void *memory) noexcept { return; }

Dims MockExecutionContext::getStrides(int bindingIndex) const noexcept { return Dims(); }

bool MockExecutionContext::setOptimizationProfile(int profileIndex) noexcept { return true; }

int MockExecutionContext::getOptimizationProfile() const noexcept { return 0; }

bool MockExecutionContext::setBindingDimensions(int bindingIndex,
    Dims dimensions) noexcept { return true; }

Dims MockExecutionContext::getBindingDimensions(int bindingIndex) const
noexcept { return Dims(); }

bool MockExecutionContext::setInputShapeBinding(int bindingIndex,
    const int32_t *data) noexcept { return true; }

bool MockExecutionContext::getShapeBinding(int bindingIndex,
    int32_t *data) const noexcept { return true; }

bool MockExecutionContext::allInputDimensionsSpecified() const noexcept { return true; }

bool MockExecutionContext::allInputShapesSpecified() const noexcept { return true; }

void MockExecutionContext::setErrorRecorder(IErrorRecorder *recorder) noexcept { return; }

IErrorRecorder *MockExecutionContext::getErrorRecorder() const noexcept { return nullptr; }

bool MockExecutionContext::executeV2(void **bindings) noexcept { return true; }

bool MockExecutionContext::enqueueV2(void **bindings, cudaStream_t stream,
                                     cudaEvent_t *inputConsumed) noexcept { return true; }

};
