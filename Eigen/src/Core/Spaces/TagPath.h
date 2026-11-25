// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_TAGPATH_H
#define EIGEN_SPACES_TAGPATH_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal A quantity as a type, for lists of them. */
template <auto Q>
struct QuantityValue {
  static constexpr auto value = Q;
};

}  // namespace internal

template <typename S, auto... Qs>
class BoundSpace;

namespace internal {

/** \internal Declared, never defined: matched by a pointer to a BoundSpace (Bound.h) or a tag derived from one. */
template <typename S, auto... Qs>
std::tuple<QuantityValue<Qs>...>* boundQuantities(const BoundSpace<S, Qs...>*);

template <typename S, auto... Qs>
S* boundBase(const BoundSpace<S, Qs...>*);

/** \internal A BoundSpace, or a tag derived from one. */
template <typename T>
concept IsBoundSpace = requires(const T* t) { boundQuantities(t); };

/** \internal The quantities of the BoundSpace \a T, as a tuple of QuantityValue. */
template <typename T>
using BoundQuantities = std::remove_pointer_t<decltype(boundQuantities(std::declval<const T*>()))>;

/** \internal The unbound space of the BoundSpace \a T. */
template <typename T>
using BoundBase = std::remove_pointer_t<decltype(boundBase(std::declval<const T*>()))>;

template <typename S>
struct TagOfImpl {
  using type = S;
};
template <IsBoundSpace S>
struct TagOfImpl<S> {
  using type = BoundBase<S>;
};

/** \internal The tag a block of type \a S answers to: \a S, or the space a BoundSpace binds units to, so that a
 * path names a block whatever units it is in (design/decisions.md [leaf-quantities]).
 */
template <typename S>
using TagOf = typename TagOfImpl<S>::type;

/** \internal The CRTP type of a dense matrix or matrix expression \a T, however qualified, from derived(),
 * so that a <tt>MatrixBase<Derived></tt> passed on from a parameter of that type has it too.
 */
template <typename T>
using CrtpDerived = std::remove_cvref_t<decltype(std::declval<std::remove_cvref_t<T>&>().derived())>;

}  // namespace internal

/** \ingroup Spaces_Module
 * A dense matrix or matrix expression, possibly cv- or reference-qualified: what a
 * <tt>const MatrixBase<Derived>&</tt> parameter accepts. It constrains the forwarding
 * references of functions that store their argument, so that a plain matrix passed as an rvalue
 * is moved rather than copied.
 */
template <typename T>
concept MatrixExpression = requires { typename internal::CrtpDerived<T>; } &&
                           std::derived_from<std::remove_cvref_t<T>, MatrixBase<internal::CrtpDerived<T>>>;

namespace internal {

/** \internal A library space type itself rather than a tag derived from one, which a path can't name. */
template <typename T>
inline constexpr bool IsUntagged = false;
template <auto U, int Dimension, int MaxDimension>
inline constexpr bool IsUntagged<QLinear<U, Dimension, MaxDimension>>{true};
template <auto U, PeriodicInterval Interval, int Dimension, int MaxDimension>
inline constexpr bool IsUntagged<QPeriodic<U, Interval, Dimension, MaxDimension>>{true};
template <typename S>
inline constexpr bool IsUntagged<DualSpace<S>>{true};
template <typename... Blocks>
inline constexpr bool IsUntagged<Composite<Blocks...>>{true};
template <typename Block, int MaxCount>
inline constexpr bool IsUntagged<Repeated<Block, MaxCount>>{true};

/** \internal Number of times \a Tag appears in \a S, not looking inside repeated blocks. */
template <typename Tag, typename S>
consteval int countTag();

/** \internal countTag() summed over \a Blocks, the blocks of a Composite. */
template <typename Tag, typename... Blocks>
consteval int countTagIn(std::tuple<Blocks...>*) {
  return (countTag<Tag, Blocks>() + ... + 0);
}

template <typename Tag, typename S>
consteval int countTag() {
  if constexpr (std::is_same_v<Tag, TagOf<S>>) {
    return 1;
  } else if constexpr (IsComposite<S>) {
    return countTagIn<Tag>(static_cast<CompositeBlocks<S>*>(nullptr));
  } else {
    return 0;
  }
}

/** \internal Where a tag sits in a space, as far as is known at compile time. */
struct TagLocation {
  int offset = -1;            ///< offset, counting fixed sizes only
  bool staticOffset = false;  ///< no resizable block precedes the tag, so \c offset is exact
  bool tail = false;          ///< the tag's block ends the space
};

/** \internal \a S, or its difference space if \a Difference: the space whose sizes an operand of values or
 * of differences takes.
 */
template <bool Difference, typename S>
struct SizedAsOf {
  using type = S;
};

template <typename S>
struct SizedAsOf<true, S> {
  using type = typename DifferenceSpaceOf<S>::type;
};

template <bool Difference, typename S>
using SizedAs = typename SizedAsOf<Difference, S>::type;

template <bool Difference, typename Tag, typename S>
consteval TagLocation locateTag();

/** \internal locateTag() over \a Blocks, the blocks of a Composite, adding up the sizes before the match. */
template <bool Difference, typename Tag, typename... Blocks>
consteval TagLocation locateTagIn(std::tuple<Blocks...>*) {
  constexpr std::size_t count = sizeof...(Blocks);
  constexpr auto counts = std::array<int, count>{countTag<Tag, Blocks>()...};
  constexpr auto inner = std::array<TagLocation, count>{locateTag<Difference, Tag, Blocks>()...};
  constexpr auto sizes = std::array<int, count>{SizedAs<Difference, Blocks>::FixedValueDimension...};
  constexpr auto fixed = std::array<bool, count>{(Blocks::ValueDimensionStep == 0)...};
  int offset = 0;
  bool staticOffset = true;
  for (std::size_t i = 0; i < count; ++i) {
    if (counts[i] > 0) {
      return {offset + inner[i].offset, staticOffset && inner[i].staticOffset, inner[i].tail && i + 1 == count};
    }
    offset += sizes[i];
    staticOffset = staticOffset && fixed[i];
  }
  return {};
}

/** \internal Location of the first \a Tag within \a S, counting value coordinates, or difference
 * coordinates if \a Difference; only consulted when countTag() finds it.
 */
template <bool Difference, typename Tag, typename S>
consteval TagLocation locateTag() {
  if constexpr (std::is_same_v<Tag, TagOf<S>>) {
    return {0, true, true};
  } else if constexpr (IsComposite<S>) {
    return locateTagIn<Difference, Tag>(static_cast<CompositeBlocks<S>*>(nullptr));
  } else {
    return {};
  }
}

/** \internal Where a path of tags resolves to, as far as is known at compile time. */
struct ResolvedIndex {
  int offset = 0;            ///< first coordinate, if \c staticOffset
  int size = 0;              ///< fixed size of the resolved block (0 if resizable)
  bool resizable = false;    ///< the resolved block is resizable
  int stride = 0;            ///< nonzero: one coordinate of every repetition of a repeated block
  bool staticOffset = true;  ///< no resizable block precedes it
  bool tail = true;          ///< it ends the space
};

/** \internal resolveTags() counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename S, typename Tag, typename... Rest>
consteval ResolvedIndex resolveTagsIn() {
  static_assert(!IsUntagged<Tag>, "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}");
  static_assert(countTag<Tag, S>() > 0, "the space does not contain this tag");
  static_assert(countTag<Tag, S>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
  using Sized = SizedAs<Difference, Tag>;
  constexpr TagLocation location = locateTag<Difference, Tag, S>();
  if constexpr (sizeof...(Rest) == 0) {
    return {location.offset, Sized::FixedValueDimension, Tag::ValueDimensionStep != 0, 0, location.staticOffset,
            location.tail};
  } else if constexpr (IsRepeated<Tag>) {
    constexpr ResolvedIndex inner = resolveTagsIn<Difference, RepeatedBlock<Tag>, Rest...>();
    static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
    return {location.offset + inner.offset, 1, false, Sized::ValueDimensionStep, location.staticOffset, location.tail};
  } else {
    constexpr ResolvedIndex inner = resolveTagsIn<Difference, Tag, Rest...>();
    return {location.offset + inner.offset,
            inner.size,
            inner.resizable,
            inner.stride,
            location.staticOffset && inner.staticOffset,
            location.tail && inner.tail};
  }
}

/** \internal Where the path \a Tag, \a Rest resolves to within \a S, counting value coordinates. */
template <typename S, typename Tag, typename... Rest>
consteval ResolvedIndex resolveTags() {
  return resolveTagsIn<false, S, Tag, Rest...>();
}

/** \internal \returns the Eigen index that the compile-time resolution \a R denotes: a single index, a
 * fixed \c seqN, or a \c seq to \c last for a resizable or repeated tail block; see idx().
 */
template <ResolvedIndex R>
constexpr auto makeIndex() {
  static_assert(R.staticOffset, "this path's offset depends on the size of a resizable block; use idx(space)");
  if constexpr (R.stride != 0) {
    static_assert(R.tail, "this repeated block is not at the end of the space; use idx(space)");
    return seq(fix<R.offset>, placeholders::last, fix<R.stride>);
  } else if constexpr (R.resizable) {
    static_assert(R.tail, "this resizable block is not at the end of the space; use idx(space)");
    return seq(fix<R.offset>, placeholders::last);
  } else if constexpr (R.size == 1) {
    return Index{R.offset};
  } else {
    return seqN(fix<R.offset>, fix<R.size>);
  }
}

/** \internal \returns the offset of the unique block of type \a Tag within \a space, counting value
 * coordinates, or difference coordinates if \a Difference, and a pointer to that block (const
 * if \a S is).
 */
template <typename Tag, bool Difference = false, typename S>
auto locate(S& space) {
  using Plain = std::remove_const_t<S>;
  using TagPointer = std::conditional_t<std::is_const_v<S>, const Tag*, Tag*>;
  if constexpr (std::is_same_v<Tag, TagOf<Plain>>) {
    return std::pair<Index, TagPointer>{0, &space};
  } else {
    static_assert(!IsUntagged<Tag>, "a path names blocks by tag; derive one, such as struct Heading : RadiansPi<> {}");
    static_assert(countTag<Tag, Plain>() > 0, "the space does not contain this tag");
    static_assert(countTag<Tag, Plain>() < 2, "the tag appears more than once; qualify it with an enclosing tag");
    auto found = std::pair<Index, TagPointer>{0, nullptr};
    Index offset = 0;
    [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      (
          [&] {
            using Block = std::tuple_element_t<Idx, CompositeBlocks<Plain>>;
            auto& block = space.template block<Idx>();
            if constexpr (countTag<Tag, Block>() > 0) {
              const auto inner = locate<Tag, Difference>(block);
              found = {offset + inner.first, inner.second};
            }
            offset += rowsOf<Difference>(block);
          }(),
          ...);
    }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<Plain>>>{});
    return found;
  }
}

/** \internal locate() along a path of tags, none of them inside a repeated block. */
template <bool Difference, typename Tag, typename... Rest, typename S>
auto locatePathIn(S& space) {
  const auto [offset, tag] = locate<Tag, Difference>(space);
  if constexpr (sizeof...(Rest) == 0) {
    return std::pair{offset, tag};
  } else {
    static_assert(!IsRepeated<Tag>, "this path continues inside a repeated block");
    const auto [innerOffset, inner] = locatePathIn<Difference, Rest...>(*tag);
    return std::pair{offset + innerOffset, inner};
  }
}

/** \internal locatePathIn() counting value coordinates. */
template <typename Tag, typename... Rest, typename S>
auto locatePath(S& space) {
  return locatePathIn<false, Tag, Rest...>(space);
}

/** \internal \returns \a index shifted by \a offset, keeping any compile-time size and increment. */
template <typename IndexType>
auto shiftIndex(const IndexType& index, Index offset) {
  if constexpr (std::is_integral_v<IndexType>) {
    return Index{index + offset};
  } else {
    return seqN(index.first() + offset, index.sizeObject(), index.incrObject());
  }
}

}  // namespace internal

/** \ingroup Spaces_Module
 * A path of tags, each naming a block inside the one before: <tt>TagPath<Ownship, Position></tt>
 * is the Position block of Ownship. A bare tag is a path of one tag. Every template that takes
 * a path takes it as one argument, and one that takes several reads its arguments as a list
 * of paths, so a path of more than one tag is always spelled as a TagPath (design/decisions.md [tag-addressing]).
 *
 * Each tag must appear once where it is looked up, and lookup does not enter a Repeated block:
 * a tag inside one is reached only through the repeated block's tag, as in
 * <tt>TagPath<Fleet, Position></tt>, and is not counted against the same tag outside it. So in
 * a space of an Ownship and a Fleet of vehicles, each with a Position, the bare Position is the
 * ownship's.
 */
template <typename... Tags>
struct TagPath {
  static_assert(sizeof...(Tags) > 0, "a tag path needs at least one tag");
};

/** \ingroup Spaces_Module
 * A list of tag paths, each a tag or a TagPath, such as the blocks a measurement reads
 * (<tt>TagPaths<Position, TagPath<Sensor, Bias>></tt>).
 */
template <typename... Paths>
struct TagPaths {};

namespace internal {

/** \internal The path of onEach<>() given none: the whole space, which is repeated. */
struct WholeSpace {};

/** \internal What on() and onEach() hold of \a X: a matrix expression evaluated to its plain matrix, so that
 * nothing it refers to need outlive it, and anything else as it is.
 */
template <typename X>
struct HeldOf {
  using type = X;
};

template <MatrixExpression X>
struct HeldOf<X> {
  using type = typename CrtpDerived<X>::PlainObject;
};

/** \internal \a value placed on the block that the path \a Path names, in a space not yet known (see on()). */
template <typename X, typename Path>
struct OnPath {
  X value;  ///< the factor or matrix placed
};

/** \internal \a value placed on every repetition of the repeated block that the path \a Path names, or of
 * the whole space if \a Path is WholeSpace, in a space not yet known (see onEach()).
 */
template <typename X, typename Path>
struct OnEach {
  X value;  ///< the factor or matrix of one repetition
};

/** \internal Whether \a T places a value by tag path, as on() and onEach() do. */
template <typename T>
inline constexpr bool isPlacedByPath = false;
template <typename X, typename Path>
inline constexpr bool isPlacedByPath<OnPath<X, Path>>{true};
template <typename X, typename Path>
inline constexpr bool isPlacedByPath<OnEach<X, Path>>{true};

}  // namespace internal

/** \ingroup Spaces_Module
 * \returns \a x placed on the block that the path \a Path (a tag or a TagPath) names, in a space
 * that is given later: a factor of a product of distributions, such as <tt>on<Heading>(uniform)</tt>,
 * or a sensor's matrix columns for the block's coordinates. A matrix expression is evaluated here.
 */
template <typename Path, typename X>
internal::OnPath<typename internal::HeldOf<X>::type, Path> on(X x) {
  return {std::move(x)};
}

/** \ingroup Spaces_Module
 * \returns \a x, given for one repetition, placed on every repetition of the repeated block that
 * the path \a Path names, as many as the space given later has, such as
 * <tt>onEach<Landmarks>(landmark)</tt>, which is \f$ I \otimes P \f$ for a covariance \f$ P \f$.
 * Without a path, <tt>onEach<>(x)</tt>, it is placed on every repetition of the whole space, which
 * is then repeated.
 */
template <typename Path = internal::WholeSpace, typename X>
internal::OnEach<typename internal::HeldOf<X>::type, Path> onEach(X x) {
  return {std::move(x)};
}

namespace internal {

/** \internal The path \a P as a TagPath: a bare tag becomes a path of one tag. */
template <typename P>
struct AsTagPathImpl {
  using type = TagPath<P>;
};

template <typename... Tags>
struct AsTagPathImpl<TagPath<Tags...>> {
  using type = TagPath<Tags...>;
};

template <typename P>
using AsTagPath = typename AsTagPathImpl<P>::type;

/** \internal \returns whether the block that the tags \a R, \a Rs... name lies in the one that \a W, \a Ws...
 * name, both looked up in one space: it is that block or inside it. A tag is unique where it is
 * looked up, so a tag found inside another is the one a lookup reaches.
 */
template <typename R, typename... Rs, typename W, typename... Ws>
consteval bool tagsLieIn(TagPath<R, Rs...>*, TagPath<W, Ws...>*) {
  if constexpr (countTag<R, W>() > 0) {
    // R is W, or inside it.
    if constexpr (sizeof...(Ws) == 0) {
      return true;
    } else if constexpr (!std::is_same_v<R, W>) {
      return tagsLieIn(static_cast<TagPath<R, Rs...>*>(nullptr), static_cast<TagPath<Ws...>*>(nullptr));
    } else if constexpr (sizeof...(Rs) == 0) {
      return false;
    } else {
      return tagsLieIn(static_cast<TagPath<Rs...>*>(nullptr), static_cast<TagPath<Ws...>*>(nullptr));
    }
  } else if constexpr (countTag<W, R>() > 0 && sizeof...(Rs) > 0) {
    // W is inside R, and the read path goes on inside R.
    return tagsLieIn(static_cast<TagPath<Rs...>*>(nullptr), static_cast<TagPath<W, Ws...>*>(nullptr));
  } else {
    return false;
  }
}

/** \internal Whether the block named by the path \a Read lies in the block named by the path \a Within, each
 * a tag or a TagPath looked up in one space: it is that block or inside it.
 */
template <typename Read, typename Within>
inline constexpr bool pathLiesIn =
    tagsLieIn(static_cast<AsTagPath<Read>*>(nullptr), static_cast<AsTagPath<Within>*>(nullptr));

/** \internal \returns \a f.template operator()<Tags...>() for the tags of the path \a P (a tag or a TagPath). */
template <typename P, typename F>
decltype(auto) withPathTags(F&& f) {
  return [&]<typename... Tags>(TagPath<Tags...>*) -> decltype(auto) {
    return f.template operator()<Tags...>();
  }(static_cast<AsTagPath<P>*>(nullptr));
}

/** \internal resolveTagsIn() for the path \a P, a tag or a TagPath. */
template <typename S, typename P, bool Difference = false>
struct ResolvePath {
  static constexpr ResolvedIndex value = resolveTagsIn<Difference, S, P>();
};

template <typename S, typename... Tags, bool Difference>
struct ResolvePath<S, TagPath<Tags...>, Difference> {
  static constexpr ResolvedIndex value = resolveTagsIn<Difference, S, Tags...>();
};

/** \internal idx() along the tags of a path, counting value coordinates, or difference coordinates if
 * \a Difference.
 */
template <bool Difference, typename Tag, typename... Rest, typename S>
auto pathIdx(const S& space) {
  using Sized = SizedAs<Difference, Tag>;
  const auto [offset, tag] = locate<Tag, Difference>(space);
  if constexpr (sizeof...(Rest) == 0) {
    if constexpr (Tag::ValueDimensionStep != 0) {
      return seqN(offset, rowsOf<Difference>(*tag));
    } else if constexpr (Sized::FixedValueDimension == 1) {
      return Index{offset};
    } else {
      return seqN(offset, fix<Sized::FixedValueDimension>);
    }
  } else if constexpr (IsRepeated<Tag>) {
    constexpr ResolvedIndex inner = resolveTagsIn<Difference, RepeatedBlock<Tag>, Rest...>();
    static_assert(inner.size == 1, "a path through a repeated block must end at a single coordinate");
    return seqN(offset + inner.offset, tag->count(), fix<Sized::ValueDimensionStep>);
  } else {
    return shiftIndex(pathIdx<Difference, Rest...>(*tag), offset);
  }
}

}  // namespace internal

/** \ingroup Spaces_Module
 * \returns the index of the subspace named by the path \a Path (a tag or a TagPath) within the
 * space type \a S, from the type alone, for direct use in Eigen indexing:
 *  - a single fixed coordinate: an \c Index, as a constant expression
 *    (<tt>static_assert(idx<State, Heading>() == 3)</tt>);
 *  - a fixed block: \c seqN(fix<offset>, fix<size>);
 *  - a resizable block at the end of the space: \c seq(fix<offset>, last);
 *  - a path through a repeated block at the end of the space to one of its coordinates,
 *    such as <tt>idx<World, TagPath<Vehicles, Altitude>>()</tt>: that coordinate of every
 *    repetition, \c seq(fix<offset>, last, fix<stride>).
 *
 * Only available when no resizable block precedes the path's block, and a resizable or
 * repeated block ends the space; otherwise idx<Path>(space), which works for every path. A tag
 * that appears more than once must be qualified by an enclosing tag; one inside a repeated block
 * is reached only through it, and is not counted against the same tag outside (see TagPath).
 *
 * The overloads of idx() are told apart by their arguments: idx<S, Path>() from the type,
 * idx<Path>(space) from a space's sizes, and idx<Path>(space, element) and
 * idx<S, Path>(element) for one repetition. Each indexes values, such as a mean; diffIdx(),
 * with the same overloads, indexes differences, such as the rows of a covariance
 * (design/decisions.md [index-names]).
 */
template <Space S, typename Path>
constexpr auto idx() {
  return internal::makeIndex<internal::ResolvePath<S, Path>::value>();
}

/** \ingroup Spaces_Module
 * idx<S, Path>() of the differences of the subspace named by \a Path: its rows of a covariance.
 * It differs from idx() where a block's differences have other sizes than its values.
 */
template <Space S, typename Path>
constexpr auto diffIdx() {
  return internal::makeIndex<internal::ResolvePath<S, Path, true>::value>();
}

/** \ingroup Spaces_Module
 * \returns the index of the subspace named by the path \a Path (a tag or a TagPath) within
 * \a space, from its runtime sizes: an \c Index for a single fixed coordinate;
 * \c seqN(offset, fix<size>) for a fixed block; \c seqN(offset, size) for a resizable one; and
 * for a path through a repeated block to one of its coordinates, \c seqN(offset, count, fix<stride>).
 * It works for every path, where idx<S, Path>() needs every offset fixed by the type.
 */
template <typename Path>
auto idx(const Space auto& space) {
  return internal::withPathTags<Path>([&]<typename... Tags>() { return internal::pathIdx<false, Tags...>(space); });
}

/** \ingroup Spaces_Module
 * idx<Path>(space) of the differences of the subspace named by \a Path: its rows of a covariance.
 */
template <typename Path>
auto diffIdx(const Space auto& space) {
  return internal::withPathTags<Path>([&]<typename... Tags>() { return internal::pathIdx<true, Tags...>(space); });
}

namespace internal {

/** \internal Index within one repetition of \a RepeatedTag of the coordinates \a Inner (all of it if none),
 * counting value coordinates, or difference coordinates if \a Difference.
 */
template <bool Difference, typename RepeatedTag, typename... Inner>
consteval ResolvedIndex repetitionIndex() {
  static_assert(IsRepeated<RepeatedTag>, "idx(..., element) takes a repeated block");
  if constexpr (sizeof...(Inner) == 0) {
    return ResolvedIndex{0, SizedAs<Difference, RepeatedTag>::ValueDimensionStep, false, 0};
  } else {
    return resolveTagsIn<Difference, RepeatedBlock<RepeatedTag>, Inner...>();
  }
}

/** \internal \returns the index of the coordinates \a Inner of one repetition that starts at \a start: a
 * single index, or a fixed \c seqN
 */
template <ResolvedIndex Inner>
auto repetitionIndexAt(Index start) {
  if constexpr (Inner.size == 1) {
    return start;
  } else {
    return seqN(start, fix<Inner.size>);
  }
}

/** \internal idx(space, element), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename Path, typename S>
auto elementIdx(const S& space, Index element) {
  return withPathTags<Path>([&]<typename RepeatedTag, typename... Inner>() {
    const auto [offset, tag] = locate<RepeatedTag, Difference>(space);
    eigen_assert(element >= 0 && element < tag->count() && "repetition out of range");
    constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
    return repetitionIndexAt<inner>(offset + element * SizedAs<Difference, RepeatedTag>::ValueDimensionStep +
                                    inner.offset);
  });
}

/** \internal idx<S, Path>(element), counting value coordinates, or difference coordinates if \a Difference. */
template <bool Difference, typename S, typename Path>
auto staticElementIdx(Index element) {
  return withPathTags<Path>([&]<typename RepeatedTag, typename... Inner>() {
    constexpr ResolvedIndex outer = resolveTagsIn<Difference, S, RepeatedTag>();
    static_assert(outer.staticOffset,
                  "this block's offset depends on the size of a resizable block; use idx(space, element)");
    constexpr ResolvedIndex inner = repetitionIndex<Difference, RepeatedTag, Inner...>();
    return repetitionIndexAt<inner>(outer.offset + element * SizedAs<Difference, RepeatedTag>::ValueDimensionStep +
                                    inner.offset);
  });
}

}  // namespace internal

/** \ingroup Spaces_Module
 * \returns the index of repetition \a element of a repeated block within \a space, from the
 * space's runtime sizes. \a Path starts at the repeated block: <tt>idx<Fleet>(space, k)</tt>
 * is vehicle \a k, and <tt>idx<TagPath<Fleet, Heading>>(space, k)</tt> its heading. The
 * result is an \c Index for one coordinate, otherwise \c seqN(start, fix<size>).
 * \a element must be one of its repetitions.
 */
template <typename Path>
auto idx(const Space auto& space, Index element) {
  return internal::elementIdx<false, Path>(space, element);
}

/** \ingroup Spaces_Module
 * idx<Path>(space, element) of the differences of one repetition: its rows of a covariance.
 */
template <typename Path>
auto diffIdx(const Space auto& space, Index element) {
  return internal::elementIdx<true, Path>(space, element);
}

/** \ingroup Spaces_Module
 * idx<Path>(space, element) for the space type \a S, without a space object. Only available
 * when no resizable block precedes the repeated block, and \a element is not range-checked, since
 * the repetition count is unknown; otherwise use idx<Path>(space, element).
 */
template <Space S, typename Path>
auto idx(Index element) {
  return internal::staticElementIdx<false, S, Path>(element);
}

/** \ingroup Spaces_Module
 * idx<S, Path>(element) of the differences of one repetition.
 */
template <Space S, typename Path>
auto diffIdx(Index element) {
  return internal::staticElementIdx<true, S, Path>(element);
}

/** \ingroup Spaces_Module
 * Rows of a contiguous block of a space.
 */
struct BlockRows {
  Index start;  ///< first row
  Index size;   ///< number of rows
};

/** \ingroup Spaces_Module
 * \returns the rows of the block named by the path \a P (a tag or a TagPath) in \a space, from
 * its runtime sizes, in an operand of values such as a mean. The path must not continue inside
 * a repeated block, so the rows are contiguous.
 *
 * \sa differenceBlockRows()
 */
template <typename P, typename S>
BlockRows blockRows(const S& space) {
  return internal::withPathTags<P>([&]<typename... Tags>() {
    const auto [offset, tag] = internal::locatePathIn<false, Tags...>(space);
    return BlockRows{offset, valueDimension(*tag)};
  });
}

/** \ingroup Spaces_Module
 * blockRows() in an operand of differences, such as the rows of a covariance.
 */
template <typename P, typename S>
BlockRows differenceBlockRows(const S& space) {
  return internal::withPathTags<P>([&]<typename... Tags>() {
    const auto [offset, tag] = internal::locatePathIn<true, Tags...>(space);
    return BlockRows{offset, differenceDimension(*tag)};
  });
}

}  // namespace Eigen

#endif  // EIGEN_SPACES_TAGPATH_H
