// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_DUAL_H
#define EIGEN_SPACES_DUAL_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup Spaces_Module
 * \brief The dual of the space \a S: the same blocks and tags, every unit inverted
 * (design/decisions.md [dual-spaces]).
 *
 * Entry (i, j) of a matrix over the spaces (R, C) has the unit of coordinate i of R times that of
 * coordinate j of C, so a linear map from X to Z is over (Z, Dual<X>), a covariance of x over (X, X), and an
 * information matrix over (Dual<X>, Dual<X>). A dual keeps the structure of its space, so a tag path into X names
 * the same block of Dual<X>, and it is linear: covectors add and subtract as plain vectors, whatever the space's
 * own plus() and minus(). Use the alias Dual, which collapses a dual of a dual to the space, and the dual of a space
 * whose every unit is dimensionless to the space itself.
 */
template <Space S>
class DualSpace {
 public:
  /** The space this is the dual of. */
  using SpaceType = S;

  static constexpr int FixedValueDimension = S::FixedValueDimension;  ///< the space's
  static constexpr int ValueDimensionStep = S::ValueDimensionStep;    ///< the space's
  static constexpr int MaxValueDimension = S::MaxValueDimension;      ///< the space's

  DualSpace() = default;

  /** The dual of \a space, with its sizes. */
  explicit DualSpace(S space) : m_space{std::move(space)} {}

  /** \returns the space this is the dual of, with its sizes */
  const S& space() const { return m_space; }

  /** \returns the space, to resize it */
  S& space() { return m_space; }

  /** \returns the number of coordinates: the space's */
  Index valueDimension() const { return m_space.valueDimension(); }

  /** Resizes the space to \a n coordinates, where it has resize(). */
  void resize(Index n)
    requires requires(S& s) { s.resize(n); }
  {
    m_space.resize(n);
  }

  /** Equal when the spaces are. */
  friend bool operator==(const DualSpace&, const DualSpace&) = default;

 private:
  EIGEN_NO_UNIQUE_ADDRESS S m_space;  ///< the space, for its sizes
};

namespace internal {

/** \internal Declared, never defined: matched by a pointer to a DualSpace or a tag derived from one. */
template <typename S>
S* dualInner(const DualSpace<S>*);

/** \internal A DualSpace, or a tag derived from one. */
template <typename T>
concept IsDualSpace = requires(const T* t) { dualInner(t); };

/** \internal The space a DualSpace \a T is the dual of. */
template <typename T>
using DualInner = std::remove_pointer_t<decltype(dualInner(std::declval<const T*>()))>;

template <typename S>
consteval bool isDimensionlessSpace();

/** \internal isDimensionlessSpace() over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct AllDimensionless;

template <typename... Blocks>
struct AllDimensionless<std::tuple<Blocks...>> : std::bool_constant<(isDimensionlessSpace<Blocks>() && ...)> {};

/** \internal Whether every coordinate of \a S is dimensionless: its leaves' units, through composites, repeated
 * blocks, and duals. A space written from scratch, which declares no unit, is.
 */
template <typename S>
consteval bool isDimensionlessSpace() {
  if constexpr (IsDualSpace<S>) {
    return isDimensionlessSpace<DualInner<S>>();
  } else if constexpr (IsComposite<S>) {
    return AllDimensionless<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return isDimensionlessSpace<RepeatedBlock<S>>();
  } else {
    return isDimensionless(leafUnit<S>());
  }
}

/** \internal The type Dual<S> denotes: the space itself where it is its own dual. */
template <typename S>
struct DualOf {
  using type = std::conditional_t<isDimensionlessSpace<S>(), S, DualSpace<S>>;
};

template <IsDualSpace S>
struct DualOf<S> {
  using type = DualInner<S>;
};

template <typename S>
consteval auto firstUnit();

/** \internal The unit of the first leaf of the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct FirstUnitOf;

template <typename First, typename... Rest>
struct FirstUnitOf<std::tuple<First, Rest...>> {
  static constexpr auto value = firstUnit<First>();
};

/** \internal \returns the unit of the first leaf of \a S: a leaf's unit(), a repeated block's, a composite's first
 * block's, and the inverse through a dual. The unit of every coordinate where hasUniformUnit() holds.
 */
template <typename S>
consteval auto firstUnit() {
  if constexpr (IsDualSpace<S>) {
    return inverseUnit(firstUnit<DualInner<S>>());
  } else if constexpr (IsComposite<S>) {
    return FirstUnitOf<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return firstUnit<RepeatedBlock<S>>();
  } else {
    return leafUnit<S>();
  }
}

template <typename S>
consteval bool hasUniformUnit();

/** \internal hasUniformUnit() over the blocks in the tuple type \a Blocks: each uniform, and all equal to the first. */
template <typename Blocks>
struct BlocksAgree;

template <typename First, typename... Rest>
struct BlocksAgree<std::tuple<First, Rest...>>
    : std::bool_constant<(hasUniformUnit<First>() && ... && hasUniformUnit<Rest>()) &&
                         (unitsEqual(firstUnit<First>(), firstUnit<Rest>()) && ...)> {};

/** \internal Whether every coordinate of \a S has one unit, so that a reduction along it is well typed: a leaf
 * always, a composite when its blocks are uniform and agree, a repeated block or a dual when its block or space is.
 */
template <typename S>
consteval bool hasUniformUnit() {
  if constexpr (IsDualSpace<S>) {
    return hasUniformUnit<DualInner<S>>();
  } else if constexpr (IsComposite<S>) {
    return BlocksAgree<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return hasUniformUnit<RepeatedBlock<S>>();
  } else {
    return true;
  }
}

/** \internal Whether every coordinate of \a S has one unit. */
template <typename S>
concept HasUniformUnit = hasUniformUnit<S>();

/** \internal \returns the unit every coordinate of \a S has, or fails to compile where the blocks of a composite
 * disagree.
 */
template <typename S>
consteval auto uniformUnit() {
  static_assert(hasUniformUnit<S>(), "the block has more than one unit");
  return firstUnit<S>();
}

}  // namespace internal

/** \ingroup Spaces_Module
 * The dual of \a S (design/decisions.md [dual-spaces]): \a S itself where every unit of \a S is dimensionless, since
 * a dimensionless space is its own dual; the space \a S is the dual of, where \a S is a DualSpace; and otherwise
 * DualSpace<S>.
 */
template <Space S>
using Dual = typename internal::DualOf<S>::type;

/** \ingroup Spaces_Module
 * \returns the unit of every coordinate of the block that the path \a Path (a tag or a TagPath) names in the space
 * type \a S, inverted through a dual; a path into a composite whose blocks have different units does not compile.
 * <tt>unitAt<State, Position>()</tt> is the unit of the Position block of State, and
 * <tt>unitAt<Dual<State>, Position>()</tt> its inverse.
 */
template <Space S, typename Path>
consteval auto unitAt() {
  if constexpr (internal::IsDualSpace<S>) {
    return internal::inverseUnit(unitAt<internal::DualInner<S>, Path>());
  } else {
    [[maybe_unused]] constexpr auto resolved = internal::ResolvePath<S, Path>::value;  // the path exists in S
    return internal::uniformUnit<internal::PathTag<Path>>();
  }
}

/** \ingroup Spaces_Module
 * \returns the unit of every coordinate of the space type \a S, or fails to compile where its blocks disagree:
 * <tt>unitOf<Position>()</tt>, and <tt>unitOf<Dual<Position>>()</tt> for its inverse. A space written from scratch
 * is dimensionless.
 */
template <Space S>
consteval auto unitOf() {
  return internal::uniformUnit<S>();
}

}  // namespace Eigen

#endif  // EIGEN_SPACES_DUAL_H
