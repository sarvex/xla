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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_

#include <memory>
#include <utility>

#include "xla/pjrt/host_callback.h"
#include "xla/python/ifrt/host_callback.h"

namespace xla {
namespace ifrt {

// Wrapper of a host callback reference that uses a PjRt `xla::HostCallback`
// with XLA host send and recv. This object is expected to be passed to the
// compiler when creating `xla::ifrt::PjRtExecutable`.
class PjRtHostSendAndRecvLoadedHostCallback final
    : public llvm::RTTIExtends<PjRtHostSendAndRecvLoadedHostCallback,
                               LoadedHostCallback> {
 public:
  PjRtHostSendAndRecvLoadedHostCallback(
      Client* client, std::unique_ptr<xla::HostCallback> host_callback)
      : client_(client), host_callback_(std::move(host_callback)) {}

  const xla::HostCallback& host_callback() const { return *host_callback_; }

  // LoadedHostCallback implementation.

  Client* client() const override { return client_; }

  static char ID;  // NOLINT

 private:
  Client* client_;
  std::unique_ptr<xla::HostCallback> host_callback_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_HOST_CALLBACK_H_
