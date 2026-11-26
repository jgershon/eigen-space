// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_TYPEDBLOCKS_H
#define EIGEN_SPACES_TYPEDBLOCKS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Blocks by tag (design/decisions.md [typed-blocks]): m.block<RowPath, ColPath>(), m.middleRows<Path>(),
// m.middleCols<Path>(), and x.segment<Path>() resolve each path against the matrix's own row or column space and view
// the block typed by the subspaces named, with their units: a bound space's units carry into its blocks, and a dual's
// block is the dual of the block. A covariance's rows are a difference space, whose tagged blocks answer to their tags
// through DifferenceOf, so the same path names a block of a mean and of a covariance. The views are writable where the
// matrix is, and an assignment into one is checked and converted by the space rules.

namespace Eigen {

namespace internal {

/** \internal The position among \a Blocks of the first block that holds \a Tag, or their count. */
template <typename Tag, typename... Blocks>
consteval std::size_t blockHoldingTag(std::tuple<Blocks...>*) {
  constexpr int counts[] = {countTag<Tag, Blocks>()..., 0};
  for (std::size_t i = 0; i < sizeof...(Blocks); ++i) {
    if (counts[i] > 0) {
      return i;
    }
  }
  return sizeof...(Blocks);
}

/** \internal The coordinates of the first \a N of \a Blocks, all of fixed size. */
template <std::size_t N, typename... Blocks>
consteval int coordinatesBefore(std::tuple<Blocks...>*) {
  constexpr int sizes[] = {Blocks::FixedValueDimension..., 0};
  int sum = 0;
  for (std::size_t i = 0; i < N; ++i) {
    sum += sizes[i];
  }
  return sum;
}

/** \internal The slots, the leaves a BoundSpace binds, of the first \a N of \a Blocks. */
template <std::size_t N, typename... Blocks>
consteval int slotsBefore(std::tuple<Blocks...>*) {
  constexpr int slots[] = {slotCount<Blocks>()..., 0};
  int sum = 0;
  for (std::size_t i = 0; i < N; ++i) {
    sum += slots[i];
  }
  return sum;
}

/** \internal \a T with the quantities of the slots \a Start onward of the tuple of QuantityValue \a Qs bound to it,
 * unless it is bound already or has no leaves.
 */
template <typename T, typename Qs, int Start, typename Slots = std::make_index_sequence<std::size_t(slotCount<T>())>>
struct BindSlotRange;

template <typename T, typename Qs, int Start, std::size_t... Slot>
struct BindSlotRange<T, Qs, Start, std::index_sequence<Slot...>> {
  using type = std::conditional_t<IsBoundSpace<T> || sizeof...(Slot) == 0, T,
                                  BoundSpace<T, std::tuple_element_t<Start + int(Slot), Qs>::value...>>;
};

/** \internal The block of the space \a S that answers to \a Tag: \c type, the block with the units \a S gives it;
 * \c offset, its first coordinate; and \c slot, its first leaf.
 */
template <typename S, typename Tag>
struct TagBlock;

template <typename S, typename Tag>
  requires std::is_same_v<TagOf<S>, Tag>
struct TagBlock<S, Tag> {
  using type = S;
  static constexpr int offset = 0;
  static constexpr int slot = 0;
};

template <typename S, typename Tag>
  requires(!std::is_same_v<TagOf<S>, Tag> && IsDualSpace<S>)
struct TagBlock<S, Tag> {
  using Inner = TagBlock<DualInner<S>, Tag>;
  using type = Dual<typename Inner::type>;
  static constexpr int offset = Inner::offset;
  static constexpr int slot = Inner::slot;
};

template <typename S, typename Tag>
  requires(!std::is_same_v<TagOf<S>, Tag> && !IsDualSpace<S> && IsBoundSpace<S>)
struct TagBlock<S, Tag> {
  using Inner = TagBlock<BoundBase<S>, Tag>;
  using type = typename BindSlotRange<typename Inner::type, BoundQuantities<S>, Inner::slot>::type;
  static constexpr int offset = Inner::offset;
  static constexpr int slot = Inner::slot;
};

template <typename S, typename Tag>
  requires(!std::is_same_v<TagOf<S>, Tag> && !IsDualSpace<S> && !IsBoundSpace<S> && IsComposite<S>)
struct TagBlock<S, Tag> {
  using Blocks = CompositeBlocks<S>;
  static_assert(!IsUntagged<Tag>, "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}");
  static_assert(countTag<Tag, S>() > 0, "the space does not contain this tag");
  static_assert(countTag<Tag, S>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
  static constexpr std::size_t index = blockHoldingTag<Tag>(static_cast<Blocks*>(nullptr));
  using Inner = TagBlock<std::tuple_element_t<index, Blocks>, Tag>;
  using type = typename Inner::type;
  static constexpr int offset = coordinatesBefore<index>(static_cast<Blocks*>(nullptr)) + Inner::offset;
  static constexpr int slot = slotsBefore<index>(static_cast<Blocks*>(nullptr)) + Inner::slot;
};

/** \internal The block of the space \a S that the path \a Path names: a tag, a TagPath, or WholeSpace. */
template <typename S, typename Path>
struct PathBlock : TagBlock<S, Path> {};

template <typename S>
struct PathBlock<S, WholeSpace> {
  using type = S;
  static constexpr int offset = 0;
};

template <typename S>
struct PathBlock<S, TagPath<>> : PathBlock<S, WholeSpace> {};

template <typename S, typename Tag, typename... Rest>
struct PathBlock<S, TagPath<Tag, Rest...>> {
  using First = TagBlock<S, Tag>;
  using Inner = PathBlock<typename First::type, TagPath<Rest...>>;
  using type = typename Inner::type;
  static constexpr int offset = First::offset + Inner::offset;
};

/** \internal The identity on coefficients, by reference, reporting the spaces \a RowSpace and \a ColSpace: the
 * writable view of a block by tag.
 */
template <typename RowSpace, typename ColSpace, typename Scalar>
struct space_view_op {
  using result_type = Scalar;
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE const Scalar& operator()(const Scalar& a) const { return a; }
  EIGEN_DEVICE_FUNC constexpr EIGEN_STRONG_INLINE Scalar& operator()(Scalar& a) const { return a; }
};

template <typename RowSpace, typename ColSpace, typename Scalar>
struct functor_traits<space_view_op<RowSpace, ColSpace, Scalar>> {
  enum { Cost = 0, PacketAccess = false };
};

template <typename RowSpace_, typename ColSpace_, typename Scalar, typename Xpr, typename StrideType>
struct spaces_of<CwiseUnaryView<space_view_op<RowSpace_, ColSpace_, Scalar>, Xpr, StrideType>> {
  using RowSpace = RowSpace_;
  using ColSpace = ColSpace_;
};

/** \internal The blocks that \a RowPath and \a ColPath name in the spaces of \a Derived. */
template <typename RowPath, typename ColPath, typename Derived>
struct TypedBlockOf {
  using RowSpace = typename spaces_of<std::remove_const_t<Derived>>::RowSpace;
  using ColSpace = typename spaces_of<std::remove_const_t<Derived>>::ColSpace;
  static_assert(FixedSizeSpace<RowSpace> && FixedSizeSpace<ColSpace>,
                "a block by tag needs spaces of fixed size, whose offsets the type fixes");
  using Rows = PathBlock<RowSpace, RowPath>;
  using Cols = PathBlock<ColSpace, ColPath>;
  static constexpr int rows = valueDimensionAtCompileTime<typename Rows::type>;
  static constexpr int cols = valueDimensionAtCompileTime<typename Cols::type>;
  using Inner = Block<Derived, rows, cols>;
};

template <typename RowPath, typename ColPath, typename Derived>
auto typedBlock(Derived& m) {
  using Of = TypedBlockOf<RowPath, ColPath, Derived>;
  typename Of::Inner inner(m, Of::Rows::offset, Of::Cols::offset);
  using View = CwiseUnaryView<space_view_op<typename Of::Rows::type, typename Of::Cols::type, typename Derived::Scalar>,
                              typename Of::Inner>;
  return View(inner);
}

template <typename RowPath, typename ColPath, typename Derived>
auto typedConstBlock(const Derived& m) {
  using Of = TypedBlockOf<RowPath, ColPath, const Derived>;
  const typename Of::Inner inner(m, Of::Rows::offset, Of::Cols::offset);
  return relabelSpaces<typename Of::Rows::type, typename Of::Cols::type>(inner);
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_TYPEDBLOCKS_H
