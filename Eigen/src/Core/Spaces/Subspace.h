// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_SUBSPACE_H
#define EIGEN_SPACES_SUBSPACE_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Subspaces, and growing and shrinking blocks (design/decisions.md [several-resizable-blocks]): the space a path
// names and the space of the coordinates outside it, for marginals and conditionals, and the rows that an
// insertion or erasure adds to or removes from an elementwise block.

namespace Eigen {

namespace internal {

/** \internal The last tag of the path \a P: a bare tag is its own. */
template <typename P>
struct PathTagOf {
  using type = P;
};

template <typename... Tags>
struct PathTagOf<TagPath<Tags...>> {
  using type = std::tuple_element_t<sizeof...(Tags) - 1, std::tuple<Tags...>>;
};

/** \internal The type of the block named by the path \a P: its last tag. */
template <typename P>
using PathTag = typename PathTagOf<P>::type;

/** \internal Whether the path \a P runs through a Repeated block: a tag before its last is repeated, so
 * idx<P>(space) is a strided index over every repetition rather than one block. A bare tag runs
 * through none.
 */
template <typename P>
inline constexpr bool RunsThroughRepeated = false;
template <typename... Tags>
inline constexpr bool RunsThroughRepeated<TagPath<Tags...>> = [] {
  constexpr bool repeated[] = {IsRepeated<Tags>...};
  for (std::size_t i = 0; i + 1 < sizeof...(Tags); ++i) {
    if (repeated[i]) {
      return true;
    }
  }
  return false;
}();

/** \internal \returns the block named by the path \a P in \a space, with its parameters and size. */
template <typename P, typename S>
const PathTag<P>& pathBlock(const S& space) {
  return withPathTags<P>([&]<typename... Tags>() -> const PathTag<P>& {
    const auto* block = locatePath<Tags...>(space).second;
    return *block;
  });
}

/** \internal \returns the unique block of type \a Tag within \a space, with its parameters and size. */
template <typename Tag, typename S>
const Tag& findTag(const S& space) {
  const Tag* block = locate<Tag>(space).second;
  return *block;
}

/** \internal \returns the space of the subspace named by \a Tag and \a Rest within \a space: the
 * tag's own block, or for a path through a repeated block, the selected coordinate
 * repeated as many times, up to the same maximum count.
 */
template <typename S, typename Tag, typename... Rest>
auto subspace(const S& space) {
  const Tag& tag = findTag<Tag>(space);
  if constexpr (sizeof...(Rest) == 0) {
    return tag;
  } else if constexpr (IsRepeated<Tag>) {
    auto inner = subspace<RepeatedBlock<Tag>, Rest...>(tag.block());
    return Repeated<decltype(inner), Tag::MaxCount>{std::move(inner), tag.count()};
  } else {
    return subspace<Tag, Rest...>(tag);
  }
}

/** \internal The space of the subspace named by \a Path within \a S, as subspace() returns it. */
template <typename S, typename... Path>
using SubspaceType = decltype(subspace<S, Path...>(std::declval<const S&>()));

/** \internal \returns the space of one repetition of the repeated block \a RepeatedTag within \a space,
 * or of the subspace named by \a Inner within it.
 */
template <typename RepeatedTag, typename... Inner, typename S>
auto elementSubspace(const S& space) {
  static_assert(IsRepeated<RepeatedTag>, "this takes a repeated block");
  const auto& block = findTag<RepeatedTag>(space).block();
  if constexpr (sizeof...(Inner) == 0) {
    return block;
  } else {
    return subspace<RepeatedBlock<RepeatedTag>, Inner...>(block);
  }
}

/** \internal \returns \a blocks as one space: the block itself if there is one, Linear<0> if there are none,
 * otherwise their Composite.
 */
template <typename... Blocks>
auto joinBlocks(std::tuple<Blocks...> blocks) {
  if constexpr (sizeof...(Blocks) == 0) {
    return Linear<0>{};
  } else if constexpr (sizeof...(Blocks) == 1) {
    return std::get<0>(std::move(blocks));
  } else {
    return std::apply([](Blocks&... b) { return Composite<Blocks...>{std::move(b)...}; }, blocks);
  }
}

template <typename Tag, typename... Rest, typename S>
auto blocksWithout(const S& space);  // defined below; blockWithout() recurses into it

/** \internal blocksWithout() for one block of a Composite: the block itself if the path's first tag is
 * not in it.
 */
template <typename Tag, typename... Rest, typename Block>
auto blockWithout(const Block& block) {
  if constexpr (countTag<Tag, Block>() > 0) {
    return blocksWithout<Tag, Rest...>(block);
  } else {
    return std::tuple<Block>{block};
  }
}

/** \internal \returns the blocks of \a space left when the subspace named by the path \a Tag, \a Rest is
 * removed, in order, as a tuple: empty if the path names all of \a space.
 *
 * A Composite that loses a block is replaced by its remaining blocks, since it no longer
 * holds what its tag names. A Repeated block that loses the same coordinates from every
 * repetition stays a Repeated block of what is left.
 */
template <typename Tag, typename... Rest, typename S>
auto blocksWithout(const S& space) {
  if constexpr (std::is_same_v<Tag, S>) {
    if constexpr (sizeof...(Rest) == 0) {
      return std::tuple<>{};
    } else if constexpr (IsRepeated<Tag>) {
      auto rest = blocksWithout<Rest...>(space.block());
      if constexpr (std::tuple_size_v<decltype(rest)> == 0) {
        return std::tuple<>{};
      } else {
        auto block = joinBlocks(std::move(rest));
        return std::tuple<Repeated<decltype(block), Tag::MaxCount>>{
            Repeated<decltype(block), Tag::MaxCount>{std::move(block), space.count()}};
      }
    } else {
      return blocksWithout<Rest...>(space);
    }
  } else {
    static_assert(!IsUntagged<Tag>, "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}");
    static_assert(IsComposite<S>, "the space does not contain this tag");
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return std::tuple_cat(blockWithout<Tag, Rest...>(space.template block<Idx>())...);
    }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<S>>>{});
  }
}

/** \internal \returns the space of the coordinates of \a space outside the subspace named by \a Given,
 * in ascending order: the remaining blocks, joined by joinBlocks().
 */
template <typename... Given, typename S>
auto complementSubspace(const S& space) {
  return joinBlocks(blocksWithout<Given...>(space));
}

/** \internal subspace() of the path \a P, a tag or a TagPath. */
template <typename P, typename S>
auto pathSubspace(const S& space) {
  return withPathTags<P>([&]<typename... Tags>() { return subspace<S, Tags...>(space); });
}

/** \internal The space of the subspace named by the path \a P within \a S. */
template <typename S, typename P>
using PathSubspaceType = decltype(pathSubspace<P>(std::declval<const S&>()));

/** \internal elementSubspace() of the path \a P, which starts at a repeated block. */
template <typename P, typename S>
auto pathElementSubspace(const S& space) {
  return withPathTags<P>([&]<typename... Tags>() { return elementSubspace<Tags...>(space); });
}

/** \internal complementSubspace() of the path \a P. */
template <typename P, typename S>
auto pathComplement(const S& space) {
  return withPathTags<P>([&]<typename... Tags>() { return complementSubspace<Tags...>(space); });
}

/** \internal A block that can grow and shrink element by element: a Repeated block (elements are repetitions)
 * or a resizable Linear or Periodic block (elements are coordinates).
 */
template <typename Block>
concept IsElementwise = IsRepeated<Block> || (IsUniformLeaf<Block> && Block::ValueDimensionStep != 0);

/** \internal \returns the number of elements of \a block: repetitions, or coordinates of a resizable leaf */
template <IsElementwise Block>
Index elementCount(const Block& block) {
  if constexpr (IsRepeated<Block>) {
    return block.count();
  } else {
    return block.valueDimension();
  }
}

/** \internal Sets the number of elements of \a block to \a count; see elementCount(). */
template <IsElementwise Block>
void setElementCount(Block& block, Index count) {
  if constexpr (IsRepeated<Block>) {
    block.setCount(count);
  } else {
    block.resize(count);
  }
}

/** \internal Rows of a block of elements in the full space. */
struct ElementRows {
  Index start;            ///< first row of values
  Index count;            ///< number of rows of values
  Index differenceStart;  ///< first row of differences
  Index differenceCount;  ///< number of rows of differences
};

/** \internal \returns the offsets of the elementwise block named by \a Path in \a space, in values and in
 * differences, and a pointer to it.
 */
template <typename Path, typename S>
auto locateElementwise(S& space) {
  auto [offset, block] = withPathTags<Path>([&]<typename... Tags>() { return locatePathIn<false, Tags...>(space); });
  const Index differenceOffset =
      withPathTags<Path>([&]<typename... Tags>() { return locatePathIn<true, Tags...>(space).first; });
  using Block = std::remove_pointer_t<decltype(block)>;
  static_assert(IsElementwise<Block>,
                "insertion and erasure take a Repeated block or a resizable Linear or Periodic block");
  return std::tuple{offset, differenceOffset, block};
}

/** \internal Grows the elementwise block named by \a Path by \a rows rows of values, inserted before its
 * element \a element. \returns the rows they occupy, in values and in differences.
 */
template <typename Path, typename S>
ElementRows growBlock(S& space, Index element, Index rows) {
  auto [offset, differenceOffset, block] = locateElementwise<Path>(space);
  using Block = std::remove_pointer_t<decltype(block)>;
  constexpr Index elementSize = Block::ValueDimensionStep;
  constexpr Index differenceElementSize = SizedAs<true, Block>::ValueDimensionStep;
  const Index count = elementCount(*block);
  eigen_assert(element >= 0 && element <= count && "insertion point out of range");
  eigen_assert(rows % elementSize == 0 && "insert whole elements");
  setElementCount(*block, count + rows / elementSize);
  return {offset + element * elementSize, rows, differenceOffset + element * differenceElementSize,
          rows / elementSize * differenceElementSize};
}

/** \internal Shrinks the elementwise block named by \a Path by \a elements elements starting at its
 * element \a element. \returns the rows they occupied, in values and in differences.
 */
template <typename Path, typename S>
ElementRows shrinkBlock(S& space, Index element, Index elements) {
  auto [offset, differenceOffset, block] = locateElementwise<Path>(space);
  using Block = std::remove_pointer_t<decltype(block)>;
  constexpr Index elementSize = Block::ValueDimensionStep;
  constexpr Index differenceElementSize = SizedAs<true, Block>::ValueDimensionStep;
  const Index count = elementCount(*block);
  eigen_assert(element >= 0 && elements >= 0 && element + elements <= count && "erased elements out of range");
  setElementCount(*block, count - elements);
  return {offset + element * elementSize, elements * elementSize, differenceOffset + element * differenceElementSize,
          elements * differenceElementSize};
}

/** \internal An index set from an index argument: a single index becomes a one-element set. */
template <typename Indices>
auto asIndexSet(const Indices& indices) {
  if constexpr (std::is_integral_v<Indices>) {
    return std::array<Index, 1>{indices};
  } else {
    return indices;
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_SUBSPACE_H
