#!/bin/bash

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

BUILD_DIR=$1

echo "Building XLA's run_hlo_module at ${BUILD_DIR}..."
echo "---------"

# Build run_hlo_module
build_start_time="$(date +%s)"
echo "run_hlo_module build start time: ${build_start_time}"
bazel --output_base=${BUILD_DIR} build \
    -c opt \
    --keep_going \
    xla/tools:run_hlo_module
build_end_time="$(date +%s)"
echo "run_hlo_module build end time: ${build_end_time}"
build_time="$((build_end_time - build_start_time))"
echo "Build time is ${build_time} seconds."
