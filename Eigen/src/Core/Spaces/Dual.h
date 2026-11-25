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

/** \internal \a Size coordinates (\c Dynamic if resizable), each measuring \a Q: one leaf of a space. */
template <auto Q, int Size>
struct LeafRun {};

/** \internal A Repeated block's leaves, repeated a number of times known at run time. */
template <typename Leaves>
struct RepeatedLeaves {};

/** \internal The leaves of the space \a S in order, as a std::tuple of LeafRun and RepeatedLeaves: through
 * composites, repeated blocks, duals (every quantity inverted), and bound spaces (the bound quantities, by position).
 */
template <typename S>
struct SpaceLeaves;

template <typename Blocks>
struct BlocksLeaves;

template <typename... Blocks>
struct BlocksLeaves<std::tuple<Blocks...>> {
  using type = decltype(std::tuple_cat(std::declval<typename SpaceLeaves<Blocks>::type>()...));
};

template <typename Leaves>
struct InvertLeaves;

template <typename Leaf>
struct InvertLeaf;

template <auto Q, int N>
struct InvertLeaf<LeafRun<Q, N>> {
  using type = LeafRun<inverseQuantity(Q), N>;
};

template <typename Leaves>
struct InvertLeaf<RepeatedLeaves<Leaves>> {
  using type = RepeatedLeaves<typename InvertLeaves<Leaves>::type>;
};

template <typename... Leaves>
struct InvertLeaves<std::tuple<Leaves...>> {
  using type = std::tuple<typename InvertLeaf<Leaves>::type...>;
};

/** \internal The leaves \a Leaves with the quantities \a Qs (a tuple of QuantityValue) put in from slot \a Offset on,
 * and the slot after them.
 */
template <typename Leaves, typename Qs, int Offset>
struct SubstituteLeaves;

template <typename Leaf, typename Qs, int Offset>
struct SubstituteLeaf;

template <auto Q, int N, typename Qs, int Offset>
struct SubstituteLeaf<LeafRun<Q, N>, Qs, Offset> {
  using type = LeafRun<std::tuple_element_t<Offset, Qs>::value, N>;
  static constexpr int next = Offset + 1;
};

template <typename Leaves, typename Qs, int Offset>
struct SubstituteLeaf<RepeatedLeaves<Leaves>, Qs, Offset> {
  using Inner = SubstituteLeaves<Leaves, Qs, Offset>;
  using type = RepeatedLeaves<typename Inner::type>;
  static constexpr int next = Inner::next;
};

template <typename Qs, int Offset>
struct SubstituteLeaves<std::tuple<>, Qs, Offset> {
  using type = std::tuple<>;
  static constexpr int next = Offset;
};

template <typename First, typename... Rest, typename Qs, int Offset>
struct SubstituteLeaves<std::tuple<First, Rest...>, Qs, Offset> {
  using Head = SubstituteLeaf<First, Qs, Offset>;
  using Tail = SubstituteLeaves<std::tuple<Rest...>, Qs, Head::next>;
  using type =
      decltype(std::tuple_cat(std::declval<std::tuple<typename Head::type>>(), std::declval<typename Tail::type>()));
  static constexpr int next = Tail::next;
};

template <typename S>
struct SpaceLeaves {
  using type = std::tuple<LeafRun<leafQuantity<S>(), valueDimensionAtCompileTime<S>>>;
};

template <typename S>
  requires IsBoundSpace<S>
struct SpaceLeaves<S> {
  using type = typename SubstituteLeaves<typename SpaceLeaves<BoundBase<S>>::type, BoundQuantities<S>, 0>::type;
};

template <typename S>
  requires(!IsBoundSpace<S> && IsDualSpace<S>)
struct SpaceLeaves<S> {
  using type = typename InvertLeaves<typename SpaceLeaves<DualInner<S>>::type>::type;
};

template <typename S>
  requires(!IsBoundSpace<S> && !IsDualSpace<S> && IsComposite<S>)
struct SpaceLeaves<S> : BlocksLeaves<CompositeBlocks<S>> {};

template <typename S>
  requires(!IsBoundSpace<S> && !IsDualSpace<S> && IsRepeated<S>)
struct SpaceLeaves<S> {
  using type = std::tuple<RepeatedLeaves<typename SpaceLeaves<RepeatedBlock<S>>::type>>;
};

/** \internal The quantities of the leaves \a Leaves in order, a repetition's once, as a tuple of QuantityValue. */
template <typename Leaves>
struct LeafQuantities;

template <typename... Leaves>
struct LeafQuantities<std::tuple<Leaves...>> {
  template <typename Leaf>
  struct Of;
  template <auto Q, int N>
  struct Of<LeafRun<Q, N>> {
    using type = std::tuple<QuantityValue<Q>>;
  };
  template <typename Inner>
  struct Of<RepeatedLeaves<Inner>> {
    using type = typename LeafQuantities<Inner>::type;
  };
  using type = decltype(std::tuple_cat(std::declval<typename Of<Leaves>::type>()...));
};

/** \internal What each slot of \a S measures, in order: the quantities a BoundSpace of \a S would hold. */
template <typename S>
using SlotQuantities = typename LeafQuantities<typename SpaceLeaves<S>::type>::type;

template <typename S>
consteval int slotCount() {
  return int(std::tuple_size_v<SlotQuantities<S>>);
}

/** \internal The slots of the block of type \a Tag in \a S, at any depth, and how many blocks of the type there are:
 * a binding by tag, or a path, names exactly one.
 */
struct SlotRange {
  int offset = 0;  ///< the block's first slot
  int count = 0;   ///< its slots
  int found = 0;   ///< how many blocks of the type there are
};

template <typename S, typename Tag>
consteval SlotRange slotRangeOf();

template <typename Tag, typename Blocks>
struct BlocksSlotRange;

template <typename Tag, typename... Blocks>
struct BlocksSlotRange<Tag, std::tuple<Blocks...>> {
  static consteval SlotRange get() {
    SlotRange result;
    int offset = 0;
    (
        [&] {
          const SlotRange inner = slotRangeOf<Blocks, Tag>();
          if (inner.found > 0 && result.found == 0) {
            result.offset = offset + inner.offset;
            result.count = inner.count;
          }
          result.found += inner.found;
          offset += slotCount<Blocks>();
        }(),
        ...);
    return result;
  }
};

template <typename S, typename Tag>
consteval SlotRange slotRangeOf() {
  if constexpr (std::is_same_v<S, Tag>) {
    return SlotRange{0, slotCount<S>(), 1};
  } else if constexpr (IsBoundSpace<S>) {
    return slotRangeOf<BoundBase<S>, Tag>();
  } else if constexpr (IsDualSpace<S>) {
    return slotRangeOf<DualInner<S>, Tag>();
  } else if constexpr (IsComposite<S>) {
    return BlocksSlotRange<Tag, CompositeBlocks<S>>::get();
  } else if constexpr (IsRepeated<S>) {
    return slotRangeOf<RepeatedBlock<S>, Tag>();
  } else {
    return SlotRange{};
  }
}

/** \internal Whether the slots \a First to \a First + \a Count of \a Qs measure one quantity in one unit. */
template <typename Qs, int First, std::size_t... Slot>
consteval bool uniformSlots(std::index_sequence<Slot...>) {
  if constexpr (sizeof...(Slot) == 0) {
    return true;
  } else {
    constexpr auto first = std::tuple_element_t<First, Qs>::value;
    return (sameQuantity(first, std::tuple_element_t<First + int(Slot), Qs>::value) && ...);
  }
}

/** \internal Whether every coordinate of \a S is dimensionless, a plain number: through composites, repeated blocks,
 * duals, and bound spaces. A space written from scratch, which declares no quantity, is.
 */
template <typename Qs>
struct AllDimensionless;

template <typename... Qs>
struct AllDimensionless<std::tuple<Qs...>> : std::bool_constant<(isDimensionless(Qs::value) && ...)> {};

template <typename S>
consteval bool isDimensionlessSpace() {
  return AllDimensionless<SlotQuantities<S>>::value;
}

/** \internal Whether every leaf of \a S has a unit, as the spaces of a QMatrix must [leaf-quantities]. */
template <typename Qs>
struct AllBound;

template <typename... Qs>
struct AllBound<std::tuple<Qs...>> : std::bool_constant<(isBound(Qs::value) && ...)> {};

template <typename S>
consteval bool isBoundSpace() {
  return AllBound<SlotQuantities<S>>::value;
}

/** \internal Whether every coordinate of \a S holds a \a Scalar of the right character [complex-scalars]. */
template <typename Scalar, typename Qs>
struct AllHoldScalar;

template <typename Scalar, typename... Qs>
struct AllHoldScalar<Scalar, std::tuple<Qs...>> : std::bool_constant<(scalarFitsQuantity<Scalar>(Qs::value) && ...)> {};

template <typename S, typename Scalar>
consteval bool spaceHoldsScalar() {
  return AllHoldScalar<Scalar, SlotQuantities<S>>::value;
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

/** \internal \returns what the first leaf of \a S measures, dimensionless where it has none: what every coordinate
 * measures where hasUniformQuantity() holds.
 */
template <typename S>
consteval auto firstQuantity() {
  using Qs = SlotQuantities<S>;
  if constexpr (std::tuple_size_v<Qs> == 0) {
    return dimensionless;
  } else {
    return std::tuple_element_t<0, Qs>::value;
  }
}

/** \internal Whether every coordinate of \a S measures one quantity in one unit, so that a reduction along it is well
 * typed.
 */
template <typename S>
consteval bool hasUniformQuantity() {
  using Qs = SlotQuantities<S>;
  return uniformSlots<Qs, 0>(std::make_index_sequence<std::tuple_size_v<Qs>>{});
}

/** \internal Whether every coordinate of \a S measures one quantity in one unit. */
template <typename S>
concept HasUniformQuantity = hasUniformQuantity<S>();

/** \internal \returns what every coordinate of \a S measures, or fails to compile where its leaves disagree. */
template <typename S>
consteval auto uniformQuantity() {
  static_assert(hasUniformQuantity<S>(), "the block measures more than one quantity, or in more than one unit");
  return firstQuantity<S>();
}

/** \internal Whether the binding \a B, by tag, names the slot \a Slot of \a S. */
template <typename S, int Slot, typename B>
consteval bool tagBindingNames() {
  if constexpr (IsTagBinding<B>) {
    constexpr SlotRange range = slotRangeOf<S, typename B::TagType>();
    static_assert(range.found == 1, "a binding by tag must name exactly one block of the space");
    return Slot >= range.offset && Slot < range.offset + range.count;
  } else {
    return false;
  }
}

/** \internal Whether the unit \a B, a binding by dimension, fits \a Leaf, what a slot without a unit measures. */
template <auto Leaf, auto B>
consteval bool unitBindingFits() {
  if constexpr (IsTagBinding<decltype(B)> || IsEigenDimensionless<decltype(Leaf)>) {
    return false;
  } else if constexpr (isBound(Leaf)) {
    return false;
  } else {
    return units_backend<decltype(Leaf)>::fits(Leaf, B);
  }
}

/** \internal The unit of the first binding by tag in \a Bs that names the slot \a Slot of \a S. */
template <typename S, int Slot, auto B, auto... Bs>
consteval auto tagBoundUnit() {
  if constexpr (tagBindingNames<S, Slot, decltype(B)>()) {
    return decltype(B)::unit;
  } else {
    return tagBoundUnit<S, Slot, Bs...>();
  }
}

/** \internal The first binding by dimension in \a Bs that fits \a Leaf. */
template <auto Leaf, auto B, auto... Bs>
consteval auto dimensionBoundUnit() {
  if constexpr (unitBindingFits<Leaf, B>()) {
    return B;
  } else {
    return dimensionBoundUnit<Leaf, Bs...>();
  }
}

/** \internal What the slot \a Slot of \a S measures once the bindings \a Bs are applied: the unit of the one binding by
 * tag that names it; else its own quantity where it has a unit or is dimensionless; else the unit of the one binding
 * by dimension that fits it.
 */
template <typename S, int Slot, auto... Bs>
consteval auto slotBound() {
  constexpr auto leaf = std::tuple_element_t<Slot, SlotQuantities<S>>::value;
  constexpr int byTag = (0 + ... + int(tagBindingNames<S, Slot, std::remove_cvref_t<decltype(Bs)>>()));
  constexpr int byDimension = (0 + ... + int(unitBindingFits<leaf, Bs>()));
  static_assert(byTag <= 1, "two bindings by tag name the same leaf");
  if constexpr (byTag == 1) {
    static_assert(!isBound(leaf), "a binding by tag names a leaf that has a unit already");
    return units_backend<decltype(leaf)>::bind(leaf, tagBoundUnit<S, Slot, Bs...>());
  } else if constexpr (isBound(leaf)) {
    return leaf;
  } else {
    static_assert(byDimension == 1,
                  "a leaf without a unit needs exactly one unit of its dimension, or a binding by tag");
    return units_backend<decltype(leaf)>::bind(leaf, dimensionBoundUnit<leaf, Bs...>());
  }
}

template <typename S, typename Slots, auto... Bs>
struct InResolve;

template <typename S, std::size_t... Slot, auto... Bs>
struct InResolve<S, std::index_sequence<Slot...>, Bs...> {
  using type = BoundSpace<S, slotBound<S, int(Slot), Bs...>()...>;
};

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief The space \a S with units bound to its leaves (design/decisions.md [leaf-quantities]).
 *
 * A tag names a space without units, <tt>struct Position : QLinear<isq::displacement, 3> {};</tt>, and In binds them:
 *  - <tt>In<Position, si::metre></tt>: a unit for each leaf whose dimension it fits, which for a leaf is the leaf;
 *  - <tt>In<State, si::metre, si::metre / si::second></tt>: one unit per dimension, wherever the blocks are;
 *  - <tt>In<State, bind<Position>(si::metre), bind<Velocity>(si::foot / si::second)></tt>: by tag at any depth,
 *    for two units of one dimension or to be explicit.
 *
 * A leaf that has a unit already, such as a periodic one, keeps it, and a leaf without one must get exactly one. The
 * result keeps the tags and structure of \a S, and so does its difference space.
 */
template <typename S, auto... Bindings>
using In = typename internal::InResolve<S, std::make_index_sequence<internal::slotCount<S>()>, Bindings...>::type;

/** \ingroup Spaces_Module
 * The dual of \a S (design/decisions.md [dual-spaces]): \a S itself where every coordinate of \a S is dimensionless,
 * since a dimensionless space is its own dual; the space \a S is the dual of, where \a S is a DualSpace; and otherwise
 * DualSpace<S>.
 */
template <Space S>
using Dual = typename internal::DualOf<S>::type;

/** \ingroup Spaces_Module
 * \returns what every coordinate of the block that the path \a Path (a tag or a TagPath) names in the space type
 * \a S measures, inverted through a dual: an mp-units reference, unit, or spec, or Eigen's dimensionless. A path to a
 * block whose leaves measure different quantities does not compile.
 */
template <Space S, typename Path>
consteval auto quantityAt() {
  if constexpr (internal::IsDualSpace<S>) {
    return internal::inverseQuantity(quantityAt<internal::DualInner<S>, Path>());
  } else {
    [[maybe_unused]] constexpr auto resolved = internal::ResolvePath<S, Path>::value;  // the path exists in S
    constexpr internal::SlotRange range = internal::slotRangeOf<S, internal::PathTag<Path>>();
    using Qs = internal::SlotQuantities<S>;
    static_assert(internal::uniformSlots<Qs, range.offset>(std::make_index_sequence<std::size_t(range.count)>{}),
                  "the block measures more than one quantity, or in more than one unit");
    return std::tuple_element_t<range.offset, Qs>::value;
  }
}

/** \ingroup Spaces_Module
 * \returns what every coordinate of the space type \a S measures, or fails to compile where its leaves disagree.
 */
template <Space S>
consteval auto quantityOf() {
  return internal::uniformQuantity<S>();
}

/** \ingroup Spaces_Module
 * \returns the unit of every coordinate of the block that \a Path names in \a S: <tt>unitAt<StateSI, Position>()</tt>
 * is <tt>si::metre</tt>, and <tt>unitAt<Dual<StateSI>, Position>()</tt> its inverse.
 */
template <Space S, typename Path>
consteval auto unitAt() {
  constexpr auto q = quantityAt<S, Path>();
  if constexpr (internal::IsEigenDimensionless<decltype(q)>) {
    return q;
  } else {
    return internal::units_backend<decltype(q)>::unitOf(q);
  }
}

/** \ingroup Spaces_Module
 * \returns the unit of every coordinate of \a S, or fails to compile where its leaves disagree.
 */
template <Space S>
consteval auto unitOf() {
  constexpr auto q = quantityOf<S>();
  if constexpr (internal::IsEigenDimensionless<decltype(q)>) {
    return q;
  } else {
    return internal::units_backend<decltype(q)>::unitOf(q);
  }
}

}  // namespace Eigen

#endif  // EIGEN_SPACES_DUAL_H
