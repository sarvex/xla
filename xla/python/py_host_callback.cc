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

#include "xla/python/py_host_callback.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "xla/layout_util.h"
#include "xla/pjrt/host_callback.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "tfrt/concurrency/ref_count.h"  // from @tf_runtime

namespace xla {

char PyDescriptorLoadedHostCallbackMakeArgs::ID = 0;
char PyHostSendAndRecvLoadedHostCallbackMakeArgs::ID = 0;

char PyDescriptorLoadedHostCallback::ID = 0;

namespace {

StatusOr<std::vector<CpuCallback::Arg>> CreateCallbackArgs(
    absl::Span<Shape const> operand_shapes) {
  std::vector<CpuCallback::Arg> callback_args(operand_shapes.size());
  for (int i = 0; i < operand_shapes.size(); ++i) {
    Shape shape = operand_shapes[i];

    if (shape.IsArray()) {
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      callback_args[i].dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(), callback_args[i].dims.begin());
      callback_args[i].strides = ByteStridesForShape(layout);
      callback_args[i].type = shape.element_type();
      callback_args[i].size_in_bytes = ShapeUtil::ByteSizeOf(layout);
      TF_ASSIGN_OR_RETURN(callback_args[i].dtype,
                          PrimitiveTypeToDtype(shape.element_type()));
    } else if (shape.IsToken()) {
      callback_args[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token arguments to Python callbacks are supported, "
          "got %s",
          shape.ToString());
    }
  }
  return callback_args;
}

StatusOr<std::vector<CpuCallback::Result>> CreateCallbackResults(
    absl::Span<Shape const> result_shapes) {
  std::vector<CpuCallback::Result> callback_results(result_shapes.size());
  for (int i = 0; i < result_shapes.size(); ++i) {
    if (result_shapes[i].IsArray()) {
      const Shape& shape =
          result_shapes[i].has_layout()
              ? result_shapes[i]
              : LayoutUtil::GetWithDefaultLayout(result_shapes[i]);
      callback_results[i].expected_dims.resize(shape.dimensions_size());
      absl::c_copy(shape.dimensions(),
                   callback_results[i].expected_dims.begin());
      callback_results[i].expected_strides = ByteStridesForShapeInt64(shape);
      callback_results[i].type = shape.element_type();
      callback_results[i].size_in_bytes = ShapeUtil::ByteSizeOf(shape);
      callback_results[i].reversed_layout.resize(shape.dimensions_size());
      absl::c_reverse_copy(shape.layout().minor_to_major(),
                           callback_results[i].reversed_layout.begin());
    } else if (result_shapes[i].IsToken()) {
      callback_results[i].type = TOKEN;
    } else {
      return InvalidArgument(
          "Only array and token return values from Python callbacks are "
          "supported, got %s",
          result_shapes[i].ToString());
    }
  }
  return callback_results;
}

StatusOr<tsl::RCReference<ifrt::LoadedHostCallback>>
MakePyDescriptorLoadedHostCallback(
    ifrt::Client* client, PyDescriptorLoadedHostCallbackMakeArgs* args) {
  ifrt::PlatformId platform_id = client->platform_id();
  if (platform_id != GpuId() && platform_id != CpuId()) {
    return Unimplemented(
        "\"python_descriptor\" host callback is only implemented on CPU and "
        "GPU");
  }

  TF_ASSIGN_OR_RETURN(auto callback_args,
                      CreateCallbackArgs(args->operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(args->result_shapes));

  // `args->callable` will be destroyed safely with `PythonRefManager` when
  // `CpuCallback` is destroyed.
  auto cpu_callback = std::make_unique<CpuCallback>(
      std::move(args->callable), callback_args, callback_results);
  return tsl::RCReference<ifrt::LoadedHostCallback>(
      tsl::MakeRef<PyDescriptorLoadedHostCallback>(client,
                                                   std::move(cpu_callback)));
}

StatusOr<tsl::RCReference<ifrt::LoadedHostCallback>>
MakePyHostSendAndRecvLoadedHostCallback(
    ifrt::Client* client, PyHostSendAndRecvLoadedHostCallbackMakeArgs* args) {
  TF_ASSIGN_OR_RETURN(auto callback_args,
                      CreateCallbackArgs(args->operand_shapes));
  TF_ASSIGN_OR_RETURN(auto callback_results,
                      CreateCallbackResults(args->result_shapes));

  // `args->callable` will be destroyed safely with `PythonRefManager` when
  // `CpuCallback` is destroyed.
  auto cpu_callback = std::make_shared<CpuCallback>(
      std::move(args->callable), callback_args, callback_results);

  auto host_callback = std::make_unique<HostCallback>();

  auto assign_arg_info = [](absl::Span<Shape const> shapes,
                            absl::Span<uint16_t const> channel_ids,
                            std::vector<HostCallbackArgInfo>& arg_infos) {
    DCHECK_EQ(shapes.size(), channel_ids.size());
    arg_infos.reserve(shapes.size());
    for (int i = 0; i < shapes.size(); ++i) {
      HostCallbackArgInfo host_callback_arg_info;
      host_callback_arg_info.channel_id = channel_ids[i];
      const auto& shape = shapes[i];
      Shape layout =
          (shape.has_layout() ? shape
                              : LayoutUtil::GetWithDefaultLayout(shape));
      host_callback_arg_info.shape = layout;
      arg_infos.push_back(std::move(host_callback_arg_info));
    }
  };

  assign_arg_info(args->operand_shapes, args->send_channel_ids,
                  host_callback->operands);
  assign_arg_info(args->result_shapes, args->recv_channel_ids,
                  host_callback->results);

  host_callback->callback = [cpu_callback = std::move(cpu_callback)](
                                void** outputs, void** inputs) {
    return cpu_callback->PrepareAndCall(outputs, inputs);
  };
  return tsl::RCReference<ifrt::LoadedHostCallback>(
      tsl::MakeRef<xla::ifrt::PjRtHostSendAndRecvLoadedHostCallback>(
          client, std::move(host_callback)));
}

// Makes a loaded host callback by wrapping a Python callable. Supports
// "python_descriptor" and "python_host_send_and_recv" callback types.
StatusOr<tsl::RCReference<ifrt::LoadedHostCallback>> MakeLoadedHostCallback(
    ifrt::PjRtCompiler* compiler, absl::string_view type,
    std::unique_ptr<ifrt::LoadedHostCallbackMakeArgs> args) {
  static_assert(sizeof(uintptr_t) == sizeof(uint64_t),
                "Expected 64-bit pointers");

  if (type == "python_descriptor") {
    auto* make_args =
        llvm::dyn_cast<PyDescriptorLoadedHostCallbackMakeArgs>(args.get());
    if (make_args == nullptr) {
      return InvalidArgument(
          "Making a \"python_descriptor\" host callback expects "
          "PyDescriptorLoadedHostCallbackMakeArgs.");
    }
    return MakePyDescriptorLoadedHostCallback(compiler->client(), make_args);
  } else if (type == "python_host_send_and_recv") {
    auto* make_args =
        llvm::dyn_cast<PyHostSendAndRecvLoadedHostCallbackMakeArgs>(args.get());
    if (make_args == nullptr) {
      return InvalidArgument(
          "Making a \"python_host_send_and_recv\" host callback expects "
          "PyHostSendAndRecvLoadedHostCallbackMakeArgs.");
    }
    return MakePyHostSendAndRecvLoadedHostCallback(compiler->client(),
                                                   make_args);
  } else {
    return Unimplemented("Unsupported callback type: %s", type);
  }
}

}  // namespace

xla::ifrt::PjRtCompiler::MakeLoadedHostCallbackFn
GetMakeLoadedHostCallbackFn() {
  return xla::ifrt::PjRtCompiler::MakeLoadedHostCallbackFn(
      [](ifrt::PjRtCompiler* compiler, absl::string_view type,
         std::unique_ptr<ifrt::LoadedHostCallbackMakeArgs> args) {
        return MakeLoadedHostCallback(compiler, type, std::move(args));
      });
}

}  // namespace xla
