#ifndef R2I_NNAPI_H
#define R2I_NNAPI_H

#include <r2i/tflite/engine.h>
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace r2i {
namespace nnapi {
class Engine : public r2i::tflite::Engine {
 public:
  Engine ();
  ~Engine ();

 protected:
  void ConfigureDelegate(::tflite::Interpreter *interpreter)
  override;

};


}
}
#endif