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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/xla_data.pb.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {
namespace ifrt {

class PjRtClient;

class PjRtCompiler final : public llvm::RTTIExtends<PjRtCompiler, Compiler> {
 public:
  using MakeLoadedHostCallbackFn =
      absl::AnyInvocable<StatusOr<tsl::RCReference<LoadedHostCallback>>(
          PjRtCompiler*, absl::string_view,
          std::unique_ptr<LoadedHostCallbackMakeArgs>)>;

  explicit PjRtCompiler(PjRtClient* client,
                        MakeLoadedHostCallbackFn make_loaded_host_callback)
      : client_(client),
        make_loaded_host_callback_(std::move(make_loaded_host_callback)) {}

  // Compiler implementation.

  ~PjRtCompiler() override = default;

  Client* client() override;

  StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks)
      override;

  StatusOr<std::unique_ptr<LoadedExecutable>> DeserializeLoadedExecutable(
      absl::string_view serialized, CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks)
      override;

  StatusOr<tsl::RCReference<LoadedHostCallback>> MakeLoadedHostCallback(
      absl::string_view type,
      std::unique_ptr<LoadedHostCallbackMakeArgs> args) override;

  static char ID;  // NOLINT

 private:
  PjRtClient* client_;
  MakeLoadedHostCallbackFn make_loaded_host_callback_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
