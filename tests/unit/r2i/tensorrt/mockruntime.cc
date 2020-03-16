// namespace mock {
//   bool fail_runtime;
// }


bool fail_runtime = false;
bool bad_cached_engine = false;
bool incompatible_model = false;


namespace nvinfer1 {
namespace {
class MockRuntime : public IRuntime {
 public:
  MockRuntime() {}

  ~MockRuntime() {}

  ICudaEngine *deserializeCudaEngine(const void *blob, std::size_t size,
                                     IPluginFactory *pluginFactory)  noexcept {
    nvinfer1::MockCudaEngine *ptr;

    if (!bad_cached_engine && !incompatible_model) {
      ptr = new nvinfer1::MockCudaEngine();
    } else {
      ptr = nullptr;
    }

    return ptr;
  };

  void setDLACore(int dlaCore) noexcept { return; };

  int getDLACore() const noexcept { return 0; };

  int getNbDLACores() const noexcept { return 0; };

  void destroy() noexcept {
    delete this;
  };

  void setGpuAllocator(IGpuAllocator *allocator) noexcept { return; };

  void setErrorRecorder(IErrorRecorder *recorder) noexcept { return; };

  IErrorRecorder *getErrorRecorder() const noexcept { return nullptr; };

};
}
}

extern "C" TENSORRTAPI void *createInferRuntime_INTERNAL(void *logger,
    int version) {
  nvinfer1::MockRuntime *ptr;

  if (!fail_runtime) {
    ptr = new nvinfer1::MockRuntime();
  } else {
    ptr = nullptr;
  }

  return ptr;
}
