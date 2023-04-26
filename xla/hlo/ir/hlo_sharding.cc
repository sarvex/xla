/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/hlo/ir/hlo_sharding.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_op_metadata.h"
#include "xla/overflow_util.h"
#include "xla/printer.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {
using absl::StrCat;

// [3,4,5]{2,1,0} => [12,1,5]{2,1,0} => [12,5,1]{1,2,0} => [12,5,1]{1,0,2}
// => [12,5]{1,0} => [60,1]{0,1} => [60]{0}
// [3,4,5]{1,2,0} [3,4,5]{0,2,1} => [3,20]{0,1}
// [3,4,5]{2,0,1}
// [3,4,5]{1,0,2} => [12,5]{0,1}
// [3,4,5]{0,1,2}
// [1,3,1,4,1,5]{0,1,5,2,3,4} => [3,4,5,1,1,1]{0,2,1,5,4,3} =>
// [3,4,5]{0,2,1} => [3,20,1]{0,1,2} => [3,20]{0,1}
void CanonicalizeIotaDims(absl::Span<int64_t>& dims,
                          absl::Span<int>& minor_to_major) {
  DCHECK_EQ(dims.size(), minor_to_major.size());
  if (dims.size() <= 1) {
    return;
  }
  absl::InlinedVector<int, 6> old_to_new_dims(dims.size());
  // Remove all dimensions of size one.
  auto remove_one_dims = [&] {
    int new_ndims = 0;
    for (int i = 0; i < dims.size(); ++i) {
      if (dims[i] == 1) {
        old_to_new_dims[i] = -1;
      } else {
        old_to_new_dims[i] = new_ndims;
        ++new_ndims;
      }
    }
    if (new_ndims == dims.size()) {
      return false;
    }
    for (int i = 0, new_idx = 0; i < dims.size(); ++i) {
      int new_dim = old_to_new_dims[i];
      if (new_dim >= 0) {
        dims[new_dim] = dims[i];
      }

      int new_minor_to_major_dim = old_to_new_dims[minor_to_major[i]];
      if (new_minor_to_major_dim >= 0) {
        minor_to_major[new_idx] = new_minor_to_major_dim;
        ++new_idx;
        DCHECK_LE(new_idx, new_ndims);
      }
    }
    minor_to_major = minor_to_major.subspan(0, new_ndims);
    dims = dims.subspan(0, new_ndims);

    return true;
  };
  // Merge subranges of dimensions that are major to minor order into single
  // dimensions of size of their product. The merged dimension is placed at
  // the first dimension of the subrange, and the other merged dimensions
  // are set to 1, which are then removed. `remove_one`_dims is always
  // called right before this, so it can assume there is no size one dimension.
  auto merge_dims = [&] {
    bool merged = false;
    for (int i = 1, base = 0, n = dims.size(); i < n; ++i) {
      int& base_dim = minor_to_major[base];
      int& dim = minor_to_major[i];
      if (base_dim == dim + 1) {
        dims[base_dim] *= dims[dim];
        dims[dim] = 1;
        --base_dim;
        ++dim;
        merged = true;
      } else {
        base = i;
      }
    }
    return merged;
  };
  while (true) {
    bool changed = remove_one_dims();
    changed |= merge_dims();
    if (!changed) {
      break;
    }
  }
}

}  // namespace

/*static*/ HloSharding::IotaTileAssignment
HloSharding::IotaTileAssignment::Create(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> transpose_dims,
    absl::Span<const int> transpose_minor_to_major) {
  absl::InlinedVector<int64_t, kInlinedDims> canonicalized_dims(
      transpose_dims.begin(), transpose_dims.end());
  absl::InlinedVector<int, kInlinedDims> canonicalized_minor_to_major(
      transpose_minor_to_major.begin(), transpose_minor_to_major.end());
  auto dims_span = absl::MakeSpan(canonicalized_dims);
  auto minor_to_major_span = absl::MakeSpan(canonicalized_minor_to_major);
  CanonicalizeIotaDims(dims_span, minor_to_major_span);
  if (dims_span.empty()) {
    canonicalized_dims[0] = 1;
    dims_span = absl::MakeSpan(canonicalized_dims.data(), 1);
    canonicalized_minor_to_major[0] = 0;
    minor_to_major_span =
        absl::MakeSpan(canonicalized_minor_to_major.data(), 1);
  }
  return IotaTileAssignment(dims, dims_span, minor_to_major_span);
}

HloSharding::IotaTileAssignment::IotaTileAssignment(
    const IotaTileAssignment& other)
    : IotaTileAssignment(other.ndims_, other.transpose_ndims_) {
  std::memcpy(dims_, other.dims_, size_bytes());
}

HloSharding::IotaTileAssignment::IotaTileAssignment(IotaTileAssignment&& other)
    : ndims_(other.ndims_),
      transpose_ndims_(other.transpose_ndims_),
      storage_({.out_of_line = other.storage_.out_of_line}) {
  SetPointers();
  if (IsInlined()) {
    std::memcpy(dims_, other.dims_, size_bytes());
  } else {
    other.storage_.out_of_line = nullptr;
  }
}

HloSharding::IotaTileAssignment& HloSharding::IotaTileAssignment::operator=(
    const IotaTileAssignment& other) {
  if (size_bytes() != other.size_bytes()) {
    if (!IsInlined()) {
      free(storage_.out_of_line);
    }
  }
  ndims_ = other.ndims_;
  transpose_ndims_ = other.transpose_ndims_;
  if (int size = size_bytes(); size > kInlinedBytes) {
    storage_.out_of_line = malloc(size);
  }
  SetPointers();
  return *this;
}

HloSharding::IotaTileAssignment& HloSharding::IotaTileAssignment::operator=(
    IotaTileAssignment&& other) {
  using std::swap;
  swap(ndims_, other.ndims_);
  swap(transpose_ndims_, other.transpose_ndims_);
  if (IsInlined()) {
    auto* old_out_of_line = storage_.out_of_line;
    std::memcpy(dims_, other.dims_, size_bytes());
    if (!other.IsInlined()) {
      other.storage_.out_of_line = old_out_of_line;
    }
  } else {
    swap(storage_.out_of_line, other.storage_.out_of_line);
  }
  SetPointers();
  return *this;
}

HloSharding::IotaTileAssignment::IotaTileAssignment(
    absl::Span<const int64_t> dims, absl::Span<const int64_t> transpose_dims,
    absl::Span<const int> transpose_minor_to_major)
    : IotaTileAssignment(dims.size(), transpose_dims.size()) {
  DCHECK_EQ(transpose_dims.size(), transpose_minor_to_major.size());
  std::memcpy(dims_, dims.data(), ndims_ * sizeof(int64_t));
  DCHECK_EQ(num_elements(), absl::c_accumulate(transpose_dims, 1LL,
                                               std::multiplies<int64_t>()));
  std::memcpy(transpose_dims_, transpose_dims.data(),
              transpose_ndims_ * sizeof(int64_t));
  std::memcpy(transpose_minor_to_major_, transpose_minor_to_major.data(),
              transpose_ndims_ * sizeof(int));
}

HloSharding::IotaTileAssignment::IotaTileAssignment(int ndims,
                                                    int transpose_ndims)
    : ndims_(ndims),
      transpose_ndims_(transpose_ndims),
      storage_({.out_of_line = MaybeAllocateOutOfLineStorage()}) {
  SetPointers();
}

void HloSharding::IotaTileAssignment::SetPointers() {
  dims_ = static_cast<int64_t*>(IsInlined() ? storage_.inlined
                                            : storage_.out_of_line);
  DCHECK(dims_ != nullptr);
  transpose_dims_ = dims_ + ndims_;
  transpose_minor_to_major_ =
      reinterpret_cast<int*>(transpose_dims_ + transpose_ndims_);
}

void HloSharding::IotaTileAssignment::Print(Printer* printer) const {
  printer->Append("devices=[");
  AppendJoin(printer, dims(), ",");
  printer->Append("]<=[");
  AppendJoin(printer, transpose_dims(), ",");
  printer->Append("]{");
  AppendJoin(printer, transpose_minor_to_major(), ",");
  printer->Append("}");
}

int64_t HloSharding::IotaTileAssignment::value_at(
    absl::Span<const int64_t> index) const {
  DCHECK_EQ(index.size(), ndims_);
  int64_t linear_index = index[0];
  for (int64_t i = 1; i < ndims_; ++i) {
    linear_index *= dims_[i];
    linear_index += index[i];
  }
  absl::InlinedVector<int64_t, kInlinedDims> transpose_index(transpose_ndims_);
  for (int64_t i = transpose_ndims_ - 1; i >= 0; --i) {
    transpose_index[i] = linear_index % transpose_dims_[i];
    linear_index /= transpose_dims_[i];
  }
  int64_t value = transpose_index[transpose_ndims_ - 1];
  for (int64_t i = transpose_ndims_ - 2; i >= 0; --i) {
    value *= transpose_dims_[transpose_minor_to_major_[i]];
    value += transpose_index[i];
  }
  return value;
}

bool HloSharding::TileAssignment::operator==(
    const TileAssignment& other) const {
  if (iota_ && other.iota_) {
    return *iota_ == *other.iota_;
  }
  return array() == other.array();
}

template <typename... Dims>
typename std::enable_if_t<array_impl::pack_is_integral<Dims...>::value, int64_t>
HloSharding::TileAssignment::operator()(Dims... dims) const {
  DCHECK_EQ(sizeof...(dims), num_dimensions());
  std::array<int64_t, sizeof...(dims)> indexes{{static_cast<int64_t>(dims)...}};
  return operator()(indexes);
}

int64_t HloSharding::TileAssignment::operator()(
    absl::Span<const int64_t> indexes) const {
  return array_ ? (*array_)(indexes) : iota_->value_at(indexes);
}

absl::Span<const int64_t> HloSharding::TileAssignment::dimensions() const {
  return array_ ? array_->dimensions() : iota_->dims();
}

int64_t HloSharding::TileAssignment::num_dimensions() const {
  return array_ ? array_->num_dimensions() : iota_->ndims();
}

int64_t HloSharding::TileAssignment::dim(int64_t n) const {
  return array_ ? array_->dim(n) : iota_->dim(n);
}
int64_t HloSharding::TileAssignment::num_elements() const {
  return array_ ? array_->num_elements() : iota_->num_elements();
}

void HloSharding::TileAssignment::Each(
    absl::FunctionRef<void(absl::Span<const int64_t>, int64_t)> f) const {
  return array_ ? array_->Each(f) : iota_->Each(f);
}

[[nodiscard]] HloSharding::TileAssignment HloSharding::TileAssignment::Reshape(
    absl::Span<const int64_t> new_dimensions) const {
  if (iota_) {
    return TileAssignment(
        IotaTileAssignment(new_dimensions, iota_->transpose_dims(),
                           iota_->transpose_minor_to_major()),
        /*shared_array=*/nullptr);
  }
  auto reshaped = std::make_shared<Array<int64_t>>(*array_);
  reshaped->Reshape(new_dimensions);
  return TileAssignment(std::move(reshaped));
}

void HloSharding::TileAssignment::Print(Printer* printer) const {
  if (iota_) {
    iota_->Print(printer);
  } else {
    printer->Append("devices=[");
    AppendJoin(printer, array().dimensions(), ",");
    printer->Append("]");
    AppendJoin(printer, array(), ",");
  }
}

std::string HloSharding::TileAssignment::ToString() const {
  StringPrinter printer;
  Print(&printer);
  return std::move(printer).ToString();
}

bool HloSharding::TileAssignment::UsesDevice(int64_t device) const {
  return iota_ ? device < iota_->num_elements()
               : absl::c_linear_search(array(), device);
}

const Array<int64_t>& HloSharding::TileAssignment::array() const {
  MaybeMaterializeFullArray();
  return *array_;
}
const std::shared_ptr<const Array<int64_t>>&
HloSharding::TileAssignment::shared_array() const {
  MaybeMaterializeFullArray();
  return shared_array_;
}

std::shared_ptr<Array<int64_t>>
HloSharding::TileAssignment::shared_array_clone() const {
  MaybeMaterializeFullArray();
  return std::make_shared<Array<int64_t>>(*array_);
}

void HloSharding::TileAssignment::MaybeMaterializeFullArray() const {
  if (array_ == nullptr) {
    DCHECK(shared_array_ == nullptr);
    DCHECK(iota_.has_value());
    const int64_t tdims = iota_->transpose_dims().size();
    absl::InlinedVector<int64_t, 6> init_dims(tdims);
    absl::InlinedVector<int64_t, 6> transpose_perm(tdims);
    for (int i = 0; i < tdims; ++i) {
      init_dims[tdims - 1 - i] =
          iota_->transpose_dims()[iota_->transpose_minor_to_major_[i]];
      transpose_perm[tdims - 1 - i] = iota_->transpose_minor_to_major_[i];
    }
    auto full = std::make_shared<Array<int64_t>>(init_dims);
    full->FillIota(0);
    full->TransposeDimensions(transpose_perm);
    full->Reshape(iota_->dims());
    shared_array_ = std::move(full);
    array_ = shared_array_.get();
  }
}

HloSharding HloSharding::AssignDevice(int64_t device_id,
                                      absl::Span<const OpMetadata> metadata) {
  return HloSharding(device_id, metadata);
}

HloSharding HloSharding::Tile1D(const Shape& input_shape, int64_t num_tiles,
                                absl::Span<const OpMetadata> metadata) {
  CHECK_EQ(1, input_shape.rank());
  CHECK_GT(num_tiles, 1);
  absl::Span<const int64_t> dimensions(&num_tiles, 1);
  return HloSharding(TileAssignment(dimensions, dimensions, {0}),
                     /*replicate_on_last_tile_dim=*/false, metadata);
}

HloSharding HloSharding::PartialTile(
    const TileAssignment& group_tile_assignment,
    absl::Span<const absl::Span<const int64_t>> replication_groups,
    absl::Span<const OpMetadata> metadata) {
  CHECK_EQ(group_tile_assignment.num_elements(), replication_groups.size());
  if (replication_groups.size() == 1) {
    return Replicate(metadata);
  }
  std::vector<int64_t> new_tile_dims(group_tile_assignment.dimensions().begin(),
                                     group_tile_assignment.dimensions().end());
  new_tile_dims.push_back(replication_groups[0].size());
  auto new_tile_assignment = std::make_shared<Array<int64_t>>(new_tile_dims);
  new_tile_assignment->Each([&](absl::Span<const int64_t> indices,
                                int64_t* device) {
    int64_t group = group_tile_assignment(indices.first(indices.size() - 1));
    *device = replication_groups[group][indices.back()];
  });
  return PartialTile(TileAssignment(std::move(new_tile_assignment)), metadata);
}

HloSharding HloSharding::PartialTile(
    const TileAssignment& tile_assignment_last_dim_replicate,
    absl::Span<const OpMetadata> metadata) {
  if (tile_assignment_last_dim_replicate.num_dimensions() == 1 ||
      tile_assignment_last_dim_replicate.dimensions().back() ==
          tile_assignment_last_dim_replicate.num_elements()) {
    return Replicate(metadata);
  }
  if (tile_assignment_last_dim_replicate.dimensions().back() == 1) {
    auto new_tile_dims = tile_assignment_last_dim_replicate.dimensions();
    new_tile_dims.remove_suffix(1);
    return HloSharding(
        tile_assignment_last_dim_replicate.Reshape(new_tile_dims),
        /*replicate_on_last_tile_dim=*/false, metadata);
  }
  std::vector<int64_t> sorted_groups(
      tile_assignment_last_dim_replicate.num_elements());
  const int64_t group_size =
      tile_assignment_last_dim_replicate.dimensions().back();
  const int64_t num_groups =
      tile_assignment_last_dim_replicate.num_elements() / group_size;
  std::vector<int32_t> current_group_idx(num_groups, 0);
  auto get_group_id = [&](absl::Span<const int64_t> indices) {
    int64_t group_id = 0;
    for (int64_t i = 0; i < indices.size() - 1; ++i) {
      group_id *= tile_assignment_last_dim_replicate.dim(i);
      group_id += indices[i];
    }
    return group_id;
  };
  tile_assignment_last_dim_replicate.Each(
      [&](absl::Span<const int64_t> indices, const int64_t device) {
        const int64_t group_id = get_group_id(indices);
        sorted_groups[group_id * group_size + current_group_idx[group_id]++] =
            device;
      });
  for (int i = 0; i < num_groups; ++i) {
    std::sort(sorted_groups.begin() + i * group_size,
              sorted_groups.begin() + (i + 1) * group_size);
  }
  absl::c_fill(current_group_idx, 0);
  auto sorted_tile = std::make_shared<Array<int64_t>>(
      tile_assignment_last_dim_replicate.dimensions());
  sorted_tile->Each([&](absl::Span<const int64_t> indices, int64_t* device) {
    const int64_t group_id = get_group_id(indices);
    *device =
        sorted_groups[group_id * group_size + current_group_idx[group_id]++];
  });
  return HloSharding(TileAssignment(std::move(sorted_tile)),
                     /*replicate_on_last_tile_dim=*/true, metadata);
}

HloSharding HloSharding::Subgroup(
    const TileAssignment& tile_assignment,
    absl::Span<const OpSharding::Type> subgroup_types,
    absl::Span<const OpMetadata> metadata) {
  if (subgroup_types.empty()) {
    return HloSharding(tile_assignment,
                       /*replicate_on_last_tile_dim=*/false, metadata);
  }
  // If there is only one type of subgrouping and there is no tiling on data
  // dimensions, it can be canonicalized to a simple manual/replicated sharding.
  if (absl::c_all_of(
          subgroup_types,
          [&](const OpSharding::Type t) { return t == subgroup_types[0]; }) &&
      Product(tile_assignment.dimensions().subspan(
          0, tile_assignment.num_dimensions() - subgroup_types.size())) == 1) {
    if (subgroup_types[0] == OpSharding::MANUAL) {
      return Manual(metadata);
    }
    if (subgroup_types[0] == OpSharding::REPLICATED) {
      return Replicate(metadata);
    }
  }
  // Normalize the subgroups to simplify two cases:
  //   - Remove trivial dims of size 1.
  //   - Merge dims of the same type.
  //   - Sort types.
  int64_t data_dims = tile_assignment.num_dimensions() - subgroup_types.size();
  std::vector<int64_t> perm(data_dims);
  std::iota(perm.begin(), perm.end(), 0);
  // Make sure the replicate dims are at the end so that we can leverage
  // PartialTile() to sort the elements.
  struct CmpTypeRepliateLast {
    bool operator()(OpSharding::Type a, OpSharding::Type b) const {
      if (a == b) {
        return false;
      }
      if (a == OpSharding::REPLICATED) {
        return false;
      }
      if (b == OpSharding::REPLICATED) {
        return true;
      }
      return a < b;
    }
  };
  std::map<OpSharding::Type, std::vector<int64_t>, CmpTypeRepliateLast>
      type_to_dims;
  bool needs_merging = false;
  for (int64_t i = 0; i < subgroup_types.size(); ++i) {
    if (tile_assignment.dim(i + data_dims) == 1) {
      needs_merging = true;
      continue;
    }
    auto& dims = type_to_dims[subgroup_types[i]];
    needs_merging |= !dims.empty();
    dims.push_back(i + data_dims);
  }
  needs_merging |= type_to_dims.size() > 1;
  auto create_sharding = [](const TileAssignment tiles,
                            absl::Span<const OpSharding::Type> types,
                            absl::Span<const OpMetadata> metadata) {
    if (types.size() == 1 && types.back() == OpSharding::REPLICATED) {
      // Normalize to partial tile.
      return PartialTile(tiles, metadata);
    }
    if (types.size() == 1 && types.back() == OpSharding::MANUAL &&
        tiles.num_elements() == tiles.dimensions().back()) {
      // Normalize to manual.
      return Manual(metadata);
    }
    if (!types.empty() && types.back() == OpSharding::REPLICATED) {
      // If the last type is REPLICATED, we first create a partially replicated
      // sharding without other subgroups so that the elements are sorted. Then
      // we fix the subgroup types.
      HloSharding sharding = PartialTile(tiles, metadata);
      sharding.replicate_on_last_tile_dim_ = false;
      for (const OpSharding::Type type : types) {
        sharding.subgroup_types_.push_back(type);
      }
      return sharding;
    }
    return HloSharding(tiles, types, metadata);
  };
  if (needs_merging) {
    auto data_tile_shape =
        absl::Span<const int64_t>(tile_assignment.dimensions())
            .subspan(0, data_dims);
    std::vector<int64_t> merged_shape(data_tile_shape.begin(),
                                      data_tile_shape.end());
    std::vector<int64_t> transposed_shape = merged_shape;
    std::vector<OpSharding::Type> merged_types;
    for (const auto& type_dims : type_to_dims) {
      int64_t dim_size = 1;
      for (int64_t dim : type_dims.second) {
        perm.push_back(dim);
        dim_size *= tile_assignment.dim(dim);
        transposed_shape.push_back(tile_assignment.dim(dim));
      }
      merged_shape.push_back(dim_size);
      merged_types.push_back(type_dims.first);
    }
    auto new_tiles = std::make_shared<Array<int64_t>>(transposed_shape);
    new_tiles->Each([&](absl::Span<const int64_t> indices, int64_t* value) {
      std::vector<int64_t> src_indices(tile_assignment.num_dimensions(), 0);
      for (int64_t i = 0; i < indices.size(); ++i) {
        src_indices[perm[i]] = indices[i];
      }
      *value = tile_assignment(src_indices);
    });
    new_tiles->Reshape(merged_shape);
    return create_sharding(TileAssignment(std::move(new_tiles)), merged_types,
                           metadata);
  }
  return create_sharding(tile_assignment, subgroup_types, metadata);
}

HloSharding HloSharding::Tuple(const ShapeTree<HloSharding>& sub_shardings) {
  std::vector<HloSharding> flattened_list;
  flattened_list.reserve(sub_shardings.leaf_count());
  for (const auto& index_to_sharding : sub_shardings.leaves()) {
    flattened_list.push_back(index_to_sharding.second);
  }
  if (flattened_list.empty()) {
    // Empty tuple sharding ends up having no leaves, but we want to allow
    // empty tuple HLO instruction results to have sharding, so we fetch the
    // root ({}) sharding value from the ShapeTree.
    // A ShapeTree created with ShapeTree<HloSharding>(shape, init) will have
    // init as value at its root.
    flattened_list.push_back(sub_shardings.element(ShapeIndex({})));
  }
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Tuple(const Shape& tuple_shape,
                               absl::Span<const HloSharding> shardings) {
  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  for (auto& sharding : shardings) {
    CHECK(!sharding.IsTuple())
        << sharding.ToString()
        << ", tuple shape = " << ShapeUtil::HumanString(tuple_shape);
  }
  std::vector<HloSharding> flattened_list(shardings.begin(), shardings.end());
  if (!flattened_list.empty()) {
    CHECK_EQ(flattened_list.size(), RequiredLeaves(tuple_shape))
        << "Flat list has " << flattened_list.size() << ", required "
        << RequiredLeaves(tuple_shape);
  }
  return HloSharding(flattened_list);
}

HloSharding HloSharding::SingleTuple(const Shape& tuple_shape,
                                     const HloSharding& sharding) {
  CHECK(tuple_shape.IsTuple()) << ShapeUtil::HumanString(tuple_shape);
  CHECK(!sharding.IsTuple()) << sharding.ToString();
  int64_t leaf_count = RequiredLeaves(tuple_shape);
  std::vector<HloSharding> flattened_list;
  flattened_list.resize(leaf_count, sharding);
  return HloSharding(flattened_list);
}

HloSharding HloSharding::Single(const Shape& shape,
                                const HloSharding& sharding) {
  return shape.IsTuple() ? SingleTuple(shape, sharding) : sharding;
}

void HloSharding::Print(Printer* printer, bool include_metadata) const {
  if (IsTuple()) {
    CHECK(metadata_.empty());
    if (ABSL_PREDICT_FALSE(tuple_elements_.empty())) {
      printer->Append("{}");
      return;
    }
    printer->Append("{");
    tuple_elements_[0].Print(printer, include_metadata);
    for (int i = 1; i < tuple_elements_.size(); ++i) {
      if (i % 5 == 0) {
        AppendCat(printer, ", /*index=", i, "*/");
      } else {
        printer->Append(", ");
      }
      tuple_elements_[i].Print(printer, include_metadata);
    }
    printer->Append("}");
    return;
  }

  auto print_metadata = [&] {
    if (include_metadata && !metadata_.empty()) {
      printer->Append(" metadata={");
      if (metadata_.size() == 1) {
        printer->Append(OpMetadataToString(metadata_.front()));
      } else {
        AppendJoin(printer, metadata_, ", ",
                   [](Printer* printer, auto& metadata) {
                     AppendCat(printer, "{", OpMetadataToString(metadata), "}");
                   });
      }
      printer->Append("}");
    }
  };

  if (replicated_) {
    printer->Append("{replicated");
    print_metadata();
    printer->Append("}");
    return;
  }

  if (manual_) {
    printer->Append("{manual");
    print_metadata();
    printer->Append("}");
    return;
  }
  if (maximal_) {
    AppendCat(printer, "{maximal device=",
              static_cast<int64_t>(*tile_assignment_.array().begin()));
    print_metadata();
    printer->Append("}");
    return;
  }

  auto print_last_tile_dims = [&] {
    if (!subgroup_types_.empty()) {
      auto op_sharding_type_to_string = [](OpSharding::Type type) {
        switch (type) {
          case OpSharding::MANUAL:
            return "manual";
          case OpSharding::MAXIMAL:
            return "maximul";
          case OpSharding::REPLICATED:
            return "replicated";
          default:
            return "error_type.";
        }
      };
      printer->Append(" last_tile_dims={");
      AppendJoin(printer, subgroup_types_, ", ",
                 [&](Printer* printer, OpSharding::Type sharding_type) {
                   printer->Append(op_sharding_type_to_string(sharding_type));
                 });
      printer->Append("}");
    }
  };
  printer->Append("{");
  tile_assignment_.Print(printer);
  if (replicate_on_last_tile_dim_) {
    printer->Append(" last_tile_dim_replicate");
  }
  print_last_tile_dims();
  print_metadata();
  printer->Append("}");
}

std::string HloSharding::ToString(bool include_metadata) const {
  StringPrinter printer;
  Print(&printer, include_metadata);
  return std::move(printer).ToString();
}

bool HloSharding::UsesDevice(int64_t device) const {
  if (IsTuple()) {
    return absl::c_any_of(tuple_elements_, [&](const HloSharding& s) {
      return s.UsesDevice(device);
    });
  }
  return replicated_ || manual_ || tile_assignment_.UsesDevice(device);
}

std::map<int64_t, int64_t> HloSharding::UsedDevices(int64_t* count) const {
  int64_t element_count = 1;
  std::map<int64_t, int64_t> device_map;
  if (IsTuple()) {
    for (auto& tuple_element_sharding : tuple_elements()) {
      auto unique_device = tuple_element_sharding.UniqueDevice();
      if (unique_device) {
        device_map[*unique_device] += 1;
      }
    }
    element_count = tuple_elements().size();
  } else {
    auto unique_device = UniqueDevice();
    if (unique_device) {
      device_map[*unique_device] += 1;
    }
  }
  if (count != nullptr) {
    *count = element_count;
  }
  return device_map;
}

std::vector<int64_t> HloSharding::TileIndexForDevice(int64_t device) const {
  CHECK(!maximal_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  std::vector<int64_t> ret_index;
  tile_assignment_.Each([&](absl::Span<const int64_t> index, int64_t d) {
    if (d == device) {
      ret_index = {index.begin(), index.end()};
    }
  });
  CHECK(!ret_index.empty());
  ret_index.resize(TiledDataRank());
  return ret_index;
}

int64_t HloSharding::DeviceForTileIndex(absl::Span<const int64_t> index) const {
  CHECK(!replicated_);
  CHECK(!manual_);
  CHECK(!IsTuple());
  if (maximal_) {
    return *tile_assignment_.array().begin();
  }
  if (index.size() == TiledDataRank() &&
      index.size() < tile_assignment_.num_dimensions()) {
    std::vector<int64_t> first_subgroup_index(index.begin(), index.end());
    for (int64_t i = 0; i < tile_assignment_.num_dimensions() - index.size();
         ++i) {
      first_subgroup_index.push_back(0);
    }
    return tile_assignment_(first_subgroup_index);
  }
  return tile_assignment_(index);
}

std::vector<int64_t> HloSharding::TileOffsetForDevice(const Shape& shape,
                                                      int64_t device) const {
  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions_size(), 0);
  }
  CHECK_EQ(shape.dimensions_size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
  }
  return index;
}

std::vector<int64_t> HloSharding::TileLimitForDevice(const Shape& shape,
                                                     int64_t device) const {
  CHECK(!IsTuple());
  CHECK(!manual_);

  if (maximal_) {
    return std::vector<int64_t>(shape.dimensions().begin(),
                                shape.dimensions().end());
  }

  CHECK_EQ(shape.dimensions_size(), TiledDataRank());
  std::vector<int64_t> index = TileIndexForDevice(device);
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    index[i] = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
  }
  return index;
}

int64_t HloSharding::RequiredLeaves(const Shape& shape) {
  // Empty tuples (with arbitrary nesting) have no leaf nodes as far as
  // ShapeUtil and ShapeTree are concerned, but they do have a single
  // tuple_elements_ entry since we want to allow empty tuple results to
  // have sharding.
  const int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  return (leaf_count == 0) ? 1 : leaf_count;
}

Status HloSharding::CheckLeafCount(const Shape& shape) const {
  int64_t leaf_count = ShapeUtil::GetLeafCount(shape);
  if (leaf_count == 0 && tuple_elements_.size() == 1) {
    // Allow (but don't require) empty tuples to have a single sharding
    return OkStatus();
  }
  TF_RET_CHECK(leaf_count == tuple_elements_.size())
      << "Shape " << ShapeUtil::HumanString(shape) << " has " << leaf_count
      << " leaf nodes while this sharding has " << tuple_elements_.size();
  return OkStatus();
}

StatusOr<ShapeTree<HloSharding>> HloSharding::AsShapeTree(
    const Shape& shape) const {
  if (IsTuple()) {
    ShapeTree<HloSharding> result(shape, HloSharding::Replicate());
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    auto it = tuple_elements_.begin();
    for (auto& index_to_sharding : result.leaves()) {
      index_to_sharding.second = *it++;
    }
    if (ShapeUtil::IsEmptyTuple(shape)) {
      // Empty tuples have no leaves, but we want to assign them a sharding
      // anyway, so we use the root element sharding.
      *result.mutable_element(ShapeIndex({})) = *it;
    }
    return std::move(result);
  } else {
    return ShapeTree<HloSharding>(shape, *this);
  }
}

StatusOr<HloSharding> HloSharding::GetTupleSharding(const Shape& shape) const {
  if (IsTuple()) {
    TF_RETURN_IF_ERROR(CheckLeafCount(shape));
    return *this;
  }
  return SingleTuple(shape, *this);
}

HloSharding HloSharding::NormalizeTupleSharding(const Shape& shape) const {
  if (shape.IsTuple() && !IsTuple()) {
    return HloSharding::SingleTuple(shape, *this);
  }
  return *this;
}

std::optional<int64_t> HloSharding::UniqueDevice() const {
  if (IsTuple()) {
    if (tuple_elements_.empty()) {
      return std::nullopt;
    }
    std::optional<int64_t> unique_device;
    for (auto& tuple_sharding : tuple_elements_) {
      auto device = tuple_sharding.UniqueDevice();
      if (!device || (unique_device && *device != *unique_device)) {
        return std::nullopt;
      }
      unique_device = device;
    }
    return unique_device;
  }
  if (!replicated_ && maximal_) {
    return static_cast<int64_t>(*tile_assignment_.array().begin());
  }
  return std::nullopt;
}

int64_t HloSharding::GetUniqueDevice() const {
  auto device = UniqueDevice();
  CHECK(device) << "Sharding does not have a unique device: " << *this;
  return *device;
}

Status HloSharding::ValidateTuple(const Shape& shape,
                                  std::optional<int64_t> num_devices) const {
  if (!shape.IsTuple()) {
    return tsl::errors::InvalidArgument(
        StrCat("Sharding is tuple-shaped but validation shape is not."));
  }
  TF_RETURN_IF_ERROR(CheckLeafCount(shape));
  if (ShapeUtil::GetLeafCount(shape) == 0 && tuple_elements_.empty()) {
    // Empty tuples are allowed to not have sharding
    return OkStatus();
  }

  // Now we've validated the number of tuple elements, it's safe to request a
  // shape tree.
  ShapeTree<HloSharding> shape_tree = GetAsShapeTree(shape);
  for (const auto& index_to_sharding : shape_tree.leaves()) {
    Status status = index_to_sharding.second.ValidateNonTuple(
        ShapeUtil::GetSubshape(shape, index_to_sharding.first), num_devices);
    if (!status.ok()) {
      tsl::errors::AppendToMessage(
          &status, StrCat("Note: While validating sharding tuple element ",
                          index_to_sharding.first.ToString(), " which is ",
                          index_to_sharding.second.ToString()));
      return status;
    }
  }
  return OkStatus();
}

Status HloSharding::Validate(const Shape& shape,
                             std::optional<int64_t> num_devices) const {
  if (shape.IsToken()) {
    return OkStatus();
  }
  Status status = IsTuple() ? ValidateTuple(shape, num_devices)
                            : ValidateNonTuple(shape, num_devices);
  if (!status.ok()) {
    tsl::errors::AppendToMessage(
        &status, StrCat("Note: While validating sharding ", ToString(),
                        " against shape ", ShapeUtil::HumanString(shape)));
  }
  return status;
}

Status HloSharding::ValidateNonTuple(const Shape& shape,
                                     std::optional<int64_t> num_devices) const {
  if (shape.IsTuple()) {
    return tsl::errors::InvalidArgument(
        "Validation shape is a tuple but sharding is not.");
  }
  if (replicated_) {
    return OkStatus();
  }

  // All tile assignments must be less than the number of available devices and
  // unique.
  bool all_devices_seen;
  if (!tile_assignment_.iota_) {
    absl::flat_hash_set<int64_t> seen_devices;
    Status status = tile_assignment_.array().EachStatus(
        [&num_devices, &seen_devices](absl::Span<const int64_t> indices,
                                      int32_t device) {
          if (num_devices.has_value() && device >= *num_devices) {
            return tsl::errors::InvalidArgument(
                StrCat("device ", device, " > num_devices (", *num_devices,
                       ") in tile assignment"));
          } else if (seen_devices.contains(device)) {
            return tsl::errors::InvalidArgument(
                StrCat("device ", device, " is not unique in tile assignment"));
          }
          seen_devices.insert(device);
          return OkStatus();
        });
    TF_RETURN_IF_ERROR(status);
    all_devices_seen =
        !num_devices.has_value() || seen_devices.size() == *num_devices;
  } else {
    all_devices_seen = !num_devices.has_value() ||
                       tile_assignment_.iota_->num_elements() == *num_devices;
  }

  if (IsTileMaximal() || IsManual()) {
    return OkStatus();
  }

  // The tile assignment tensor must have the same rank as the tiled data rank.
  if (shape.rank() != TiledDataRank()) {
    return tsl::errors::InvalidArgument(
        "Number of tile assignment dimensions (excluding subgroups) is "
        "different than the input rank. "
        "sharding=",
        ToString(), ", input_shape=", ShapeUtil::HumanString(shape));
  }

  // All devices should be seen in the tile assignment.
  if (!all_devices_seen) {
    return tsl::errors::InvalidArgument("tile_assignment should have ",
                                        *num_devices, " devices");
  }

  // The correct constructor has to be used to create tile maximal shardings.
  if (tile_assignment_.num_elements() == 1) {
    return tsl::errors::InvalidArgument(
        "Tile assignment only contains a single device. If a replicated "
        "sharding was intended, use HloSharding::Replicated(). If a device "
        "placement was intended, use HloSharding::AssignDevice()");
  }
  return OkStatus();
}

/*static*/ StatusOr<HloSharding> HloSharding::FromProto(
    const OpSharding& proto) {
  std::vector<OpMetadata> metadata(proto.metadata().begin(),
                                   proto.metadata().end());
  std::vector<int> subgroup_types_int(proto.last_tile_dims().begin(),
                                      proto.last_tile_dims().end());
  std::vector<OpSharding::Type> subgroup_types;
  absl::c_transform(
      subgroup_types_int, std::back_inserter(subgroup_types),
      [](const int type) { return static_cast<OpSharding::Type>(type); });
  if (proto.type() == OpSharding::TUPLE) {
    TF_RET_CHECK(metadata.empty())
        << "Tuple sharding is expected to have no metadata.";
    std::vector<HloSharding> tuple_shardings;
    tuple_shardings.reserve(proto.tuple_shardings().size());
    for (const OpSharding& tuple_sharding_proto : proto.tuple_shardings()) {
      TF_ASSIGN_OR_RETURN(HloSharding sharding,
                          HloSharding::FromProto(tuple_sharding_proto));
      tuple_shardings.push_back(sharding);
    }
    return HloSharding(tuple_shardings);
  } else if (proto.type() == OpSharding::REPLICATED) {
    return Replicate(metadata);
  } else if (proto.type() == OpSharding::MANUAL) {
    return Manual(metadata);
  } else if (proto.tile_assignment_devices().size() == 1) {
    return HloSharding(proto.tile_assignment_devices(0), metadata);
  }

  TF_RET_CHECK(proto.type() != OpSharding::MAXIMAL)
      << "Maximal sharding is expected to have single device assignment, but "
      << proto.tile_assignment_devices().size() << " has provided.";

  const bool use_iota_tile_assignments = proto.iota_dimensions_size() > 0;
  if (use_iota_tile_assignments) {
    TF_RET_CHECK(proto.tile_assignment_devices().empty());
    TF_RET_CHECK(proto.iota_dimensions_size() ==
                 proto.iota_minor_to_major_size());
  } else {
    TF_RET_CHECK(proto.tile_assignment_devices().size() > 1);
  }

  TF_RET_CHECK(!proto.tile_assignment_dimensions().empty());

  auto product_no_overflow =
      [](absl::Span<const int64_t> dims) -> StatusOr<int64_t> {
    int64_t product_of_dimensions = 1;
    for (auto dimension : dims) {
      TF_RET_CHECK(dimension > 0);
      product_of_dimensions =
          MultiplyWithoutOverflow(product_of_dimensions, dimension);
      TF_RET_CHECK(product_of_dimensions > 0);
    }
    return product_of_dimensions;
  };

  // RE: the product of tile assignment tensor dimensions must be
  // equal to tile_assignment_devices.size() or the product of iota_dimensions.
  TF_ASSIGN_OR_RETURN(int64_t product_of_dimensions,
                      product_no_overflow(proto.tile_assignment_dimensions()));
  if (use_iota_tile_assignments) {
    TF_ASSIGN_OR_RETURN(int64_t product_of_iota_dimensions,
                        product_no_overflow(proto.iota_dimensions()));
    TF_RET_CHECK(product_of_dimensions == product_of_iota_dimensions);
  } else {
    TF_RET_CHECK(product_of_dimensions ==
                 proto.tile_assignment_devices().size());
  }

  TileAssignment tiles = [&] {
    if (use_iota_tile_assignments) {
      return TileAssignment(proto.tile_assignment_dimensions(),
                            proto.iota_dimensions(),
                            proto.iota_minor_to_major());
    }
    auto tiles =
        std::make_shared<Array<int64_t>>(proto.tile_assignment_dimensions());
    absl::c_copy(proto.tile_assignment_devices(), tiles->begin());
    return TileAssignment(std::move(tiles));
  }();
  if (!subgroup_types.empty()) {
    TF_RET_CHECK(!proto.replicate_on_last_tile_dim());
    return Subgroup(std::move(tiles), subgroup_types, metadata);
  }
  return proto.replicate_on_last_tile_dim()
             ? PartialTile(std::move(tiles), metadata)
             : HloSharding(std::move(tiles),
                           /*replicate_on_last_tile_dim=*/false, metadata);
}

OpSharding HloSharding::ToProto() const {
  OpSharding result;

  if (IsTuple()) {
    CHECK(metadata_.empty());
    for (const HloSharding& element : tuple_elements_) {
      *result.add_tuple_shardings() = element.ToProto();
    }
    result.set_type(OpSharding::TUPLE);
    return result;
  }

  result.mutable_metadata()->Reserve(metadata_.size());
  for (const auto& metadata : metadata_) {
    *result.add_metadata() = metadata;
  }

  result.mutable_tile_assignment_dimensions()->Reserve(
      tile_assignment_.num_dimensions());
  absl::c_copy(tile_assignment_.dimensions(),
               tsl::protobuf::RepeatedFieldBackInserter(
                   result.mutable_tile_assignment_dimensions()));

  if (tile_assignment_.iota_) {
    result.mutable_iota_dimensions()->Reserve(
        tile_assignment_.iota_->transpose_dims().size());
    absl::c_copy(tile_assignment_.iota_->transpose_dims(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_iota_dimensions()));
    result.mutable_iota_minor_to_major()->Reserve(
        tile_assignment_.iota_->transpose_minor_to_major().size());
    absl::c_copy(tile_assignment_.iota_->transpose_minor_to_major(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_iota_minor_to_major()));
  } else {
    result.mutable_tile_assignment_devices()->Reserve(
        tile_assignment_.num_elements());
    absl::c_copy(tile_assignment_.array(),
                 tsl::protobuf::RepeatedFieldBackInserter(
                     result.mutable_tile_assignment_devices()));
  }
  if (IsReplicated()) {
    result.set_type(OpSharding::REPLICATED);
    result.clear_tile_assignment_dimensions();
  } else if (IsTileMaximal()) {
    result.set_type(OpSharding::MAXIMAL);
  } else if (IsManual()) {
    result.set_type(OpSharding::MANUAL);
    result.clear_tile_assignment_dimensions();
  } else {
    result.set_type(OpSharding::OTHER);
    result.set_replicate_on_last_tile_dim(ReplicateOnLastTileDim());
    for (auto type : subgroup_types_) {
      result.add_last_tile_dims(type);
    }
  }
  return result;
}

Shape HloSharding::TileShape(const Shape& shape) const {
  if (IsTileMaximal() || IsManual()) {
    return shape;
  }
  Shape result_shape = shape;
  for (int64_t i = 0; i < TiledDataRank(); ++i) {
    result_shape.set_dimensions(
        i, CeilOfRatio<int64_t>(shape.dimensions(i), tile_assignment_.dim(i)));
  }
  return result_shape;
}

Shape HloSharding::TileShape(const Shape& shape, int64_t device) const {
  if (IsTileMaximal() || IsManual()) {
    return shape;
  }

  std::vector<int64_t> index = TileIndexForDevice(device);
  Shape result_shape = shape;
  for (int64_t i = 0; i < index.size(); ++i) {
    const int64_t shape_dim = shape.dimensions(i);
    int64_t offset = std::min(
        index[i] * CeilOfRatio(shape_dim, tile_assignment_.dim(i)), shape_dim);
    int64_t limit = std::min(
        (index[i] + 1) * CeilOfRatio(shape_dim, tile_assignment_.dim(i)),
        shape_dim);
    result_shape.set_dimensions(i, limit - offset);
  }
  return result_shape;
}

int64_t HloSharding::TotalNumTiles() const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions()));
}

int64_t HloSharding::NumTiles() const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  return Product(absl::Span<const int64_t>(tile_assignment_.dimensions())
                     .subspan(0, TiledDataRank()));
}

int64_t HloSharding::NumTiles(absl::Span<const int64_t> dims) const {
  if (IsTileMaximal()) {
    return 1;
  }
  CHECK(!IsManual());
  CHECK(!ReplicateOnLastTileDim() ||
        !absl::c_linear_search(dims, tile_assignment().num_dimensions() - 1));
  int64_t num_tiles = 1;
  for (auto d : dims) {
    CHECK(d < tile_assignment().num_dimensions());
    num_tiles *= tile_assignment().dim(d);
  }
  return num_tiles;
}

HloSharding HloSharding::GetSubSharding(const Shape& shape,
                                        const ShapeIndex& index) const {
  CHECK(IsTuple());
  int64_t sharding_index = 0;
  const Shape* sub_shape = &shape;
  for (int64_t idx : index) {
    for (int64_t i = 0; i < idx; ++i) {
      sharding_index +=
          ShapeUtil::GetLeafCount(ShapeUtil::GetSubshape(*sub_shape, {i}));
    }
    sub_shape = &ShapeUtil::GetSubshape(*sub_shape, {idx});
  }
  if (sub_shape->IsTuple()) {
    auto begin_it = tuple_elements_.begin() + sharding_index;
    std::vector<HloSharding> sub_shardings(
        begin_it, begin_it + ShapeUtil::GetLeafCount(*sub_shape));
    return HloSharding::Tuple(*sub_shape, sub_shardings);
  } else {
    return tuple_elements_[sharding_index];
  }
}

std::optional<HloSharding> HloSharding::ExtractSingleSharding() const {
  if (!IsTuple()) {
    return *this;
  }
  if (tuple_elements_.empty()) {
    return std::nullopt;
  }
  for (int64_t i = 1; i < tuple_elements_.size(); ++i) {
    if (tuple_elements_[0] != tuple_elements_[i]) {
      return std::nullopt;
    }
  }
  return tuple_elements_.front();
}

HloSharding HloSharding::WithMetadata(absl::Span<const OpMetadata> metadata,
                                      bool overwrite) const {
  auto assign_metadata = [&](HloSharding& sharding) {
    if (sharding.metadata_.empty() || overwrite) {
      sharding.metadata_.assign(metadata.begin(), metadata.end());
    }
  };

  HloSharding sharding = *this;
  if (sharding.IsTuple()) {
    for (HloSharding& sub_sharding : sharding.tuple_elements()) {
      assign_metadata(sub_sharding);
    }
  } else {
    assign_metadata(sharding);
  }
  return sharding;
}

HloSharding HloSharding::WithoutMetadata() const {
  HloSharding sharding = *this;
  sharding.metadata_.clear();
  for (HloSharding& sub_sharding : sharding.tuple_elements()) {
    sub_sharding.metadata_.clear();
  }
  return sharding;
}

std::ostream& operator<<(std::ostream& out, const HloSharding& sharding) {
  out << sharding.ToString();
  return out;
}

}  // namespace xla
