/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_HOST_CALLBACK_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_HOST_CALLBACK_H_

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/shape.h"

namespace xla {

// Wrapper of a Python function and options necessary for making a
// "python_descriptor"-type host callback.
struct PyDescriptorLoadedHostCallbackMakeArgs final
    : llvm::RTTIExtends<PyDescriptorLoadedHostCallbackMakeArgs,
                        ifrt::LoadedHostCallbackMakeArgs> {
  pybind11::function callable;
  // TODO(hyeontaek): Migrate `xla::Shape` to IFRT types.
  absl::Span<Shape const> operand_shapes;
  absl::Span<Shape const> result_shapes;

  // LoadedHostCallbackMakeArgs implementation.

  ~PyDescriptorLoadedHostCallbackMakeArgs() override = default;

  static char ID;  // NOLINT
};

// Wrapper of a Python function and options necessary for making a
// "python_host_send_and_recv"-type host callback.
struct PyHostSendAndRecvLoadedHostCallbackMakeArgs final
    : llvm::RTTIExtends<PyHostSendAndRecvLoadedHostCallbackMakeArgs,
                        ifrt::LoadedHostCallbackMakeArgs> {
  pybind11::function callable;
  // TODO(hyeontaek): Migrate `xla::Shape` to IFRT types.
  absl::Span<Shape const> operand_shapes;
  absl::Span<Shape const> result_shapes;
  absl::Span<uint16_t const> send_channel_ids;
  absl::Span<uint16_t const> recv_channel_ids;

  // LoadedHostCallbackMakeArgs implementation.

  ~PyHostSendAndRecvLoadedHostCallbackMakeArgs() override = default;

  static char ID;  // NOLINT
};

// Wrapper of a Python host callback reference that uses a JAX
// `xla::CpuCallback` and a descriptor that is a raw pointer to
// `xla::CpuCallback`. The descriptor should be passed into a
// 'xla_python_cpu_callback' or 'xla_python_gpu_callback' CustomCall as its
// first argument.
class PyDescriptorLoadedHostCallback final
    : public llvm::RTTIExtends<PyDescriptorLoadedHostCallback,
                               ifrt::LoadedHostCallback> {
 public:
  PyDescriptorLoadedHostCallback(ifrt::Client* client,
                                 std::unique_ptr<CpuCallback> cpu_callback)
      : client_(client), cpu_callback_(std::move(cpu_callback)) {}

  uint64_t descriptor() const {
    return absl::bit_cast<uint64_t>(cpu_callback_.get());
  }

  // LoadedHostCallback implementation.

  ifrt::Client* client() const override { return client_; }

  static char ID;  // NOLINT

 private:
  ifrt::Client* client_;
  std::unique_ptr<CpuCallback> cpu_callback_;
};

// Returns an `xla::ifrt::PjRtCompiler::MakeLoadedHostCallback` that makes a
// host callback from a Python function. This function can be passed to the
// constructor of `xla::ifrt::PjRtClient` and similar IFRT client
// implementations that runs the host callback on the local host.
//
// Supports "python_descriptor" and "python_host_send_and_recv" callback types.
xla::ifrt::PjRtCompiler::MakeLoadedHostCallbackFn GetMakeLoadedHostCallbackFn();

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_HOST_CALLBACK_H_
