/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

// TODO(hyeontaek): Generalize `xla::CompileOptions`.
using CompileOptions = ::xla::CompileOptions;

// Represents a compiler that creates an `Executable` that can run a computation
// on devices.
class Compiler : public llvm::RTTIExtends<Compiler, llvm::RTTIRoot> {
 public:
  virtual Client* client() = 0;

  // Compiles `mlir_module` and returns an `LoadedExecutable`.
  //
  // TODO(hyeontaek): Introduce `Platform`/`Topology` and return `Executable`
  // instead of `LoadedExecutable`. This will factor out the loading portion of
  // the compilation, enabling ahead-of-time compilation.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>>
          loaded_host_callbacks) = 0;

  // Deserializes a serialized executable as produced by
  // `LoadedExecutable::Serialize()`. The compatibility of `serialized` is
  // implementation specific.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>>
  DeserializeLoadedExecutable(absl::string_view serialized,
                              CompileOptions options,
                              std::vector<tsl::RCReference<LoadedHostCallback>>
                                  loaded_host_callbacks) = 0;

  // Makes a loaded host callback that can be called during an execution of a
  // `LoadedExecutable`.
  //
  // `type` specifies the type of the loaded host callback. `args` is an opaque
  // data that is necessary to make the loaded host callback. Their
  // interpretation is implementation specific.
  //
  // The returned `LoadedHostCallback` is an opaque reference that is expected
  // to keep the loaded host callback alive. This object can be passed to
  // `LoadedExecutable::Execute()`.
  virtual StatusOr<tsl::RCReference<LoadedHostCallback>> MakeLoadedHostCallback(
      absl::string_view type,
      std::unique_ptr<LoadedHostCallbackMakeArgs> args) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_
