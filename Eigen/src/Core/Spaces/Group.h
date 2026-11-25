// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_GROUP_H
#define EIGEN_SPACES_GROUP_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template <typename S>
consteval bool isAdditiveGroup();

/** \internal isAdditiveGroup() over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct AllAdditive;

template <typename... Blocks>
struct AllAdditive<std::tuple<Blocks...>> : std::bool_constant<(isAdditiveGroup<Blocks>() && ...)> {};

/** \internal Whether \a S is a group under addition, so that +, -, and the coefficient-wise operations on its
 * coordinates are its own operations (design/decisions.md [additive-groups]): a leaf whose values and differences
 * are plain, or a Periodic one, which adds exactly modulo its period; a Composite or Repeated of such; a dual, which
 * is linear [dual-spaces]; and a bound space if its unbound one is. The classification is that of EmrysFilter's
 * groups-prototype, where Linear and Periodic are the groups under addition; a UnitQuaternion, or a space with a plus()
 * of its own, is not one.
 */
template <typename S>
consteval bool isAdditiveGroup() {
  if constexpr (IsBoundSpace<S>) {
    return isAdditiveGroup<BoundBase<S>>();
  } else if constexpr (IsDualSpace<S>) {
    return true;
  } else if constexpr (IsComposite<S>) {
    return AllAdditive<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return isAdditiveGroup<RepeatedBlock<S>>();
  } else {
    return isLinear<S> || IsPeriodic<S>;
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_GROUP_H
