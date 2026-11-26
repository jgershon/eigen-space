// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_BOUND_H
#define EIGEN_SPACES_BOUND_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal The number of leaves of \a S in order, a Repeated's block counted once: the slots a BoundSpace gives
 * quantities to. Defined in Dual.h, with the walk over a space's leaves.
 */
template <typename S>
consteval int slotCount();

/** \internal What slot \a Slot of the difference space \a D of \a S measures once \a S is bound to \a Qs: the
 * difference's own quantity where the slot of \a S had one already, and the bound one otherwise. A rotation's
 * values are numbers and its differences radians, so a unit bound by position must not cross from one to the other.
 */
template <typename S, typename D, int Slot, auto... Qs>
consteval auto differenceSlotQuantity();

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief The space \a S with a quantity bound to each of its leaves, by position: what In builds
 * (design/decisions.md [leaf-quantities]).
 *
 * It keeps everything of \a S, its blocks, tags, sizes, and operations, and reports \a Qs, one per leaf in order, a
 * Repeated's block counted once, as what the leaves measure. Its difference space binds the same quantities to the
 * difference space of \a S, whose leaves are in the same order. Write it as In; the positional form is what a
 * binding becomes.
 */
template <typename S, auto... Qs>
class BoundSpace : public S {
 public:
  /** The space the quantities are bound to. */
  using UnboundSpace = S;

  using S::S;
  BoundSpace() = default;

  /** \a space, with its sizes, with the quantities bound. */
  explicit BoundSpace(S space) : S{std::move(space)} {}

  /** \returns the space without the quantities, with its sizes */
  const S& unbound() const { return *this; }

  /** The difference space of \a S with the same quantities. */
  auto difference() const
    requires internal::HasDifference<S>
  {
    using Difference = DifferenceSpace<S>;
    static_assert(internal::slotCount<Difference>() == sizeof...(Qs),
                  "the difference space must have the leaves of the space, in order, to keep its quantities");
    return boundDifference<Difference>(std::make_index_sequence<sizeof...(Qs)>{});
  }

  /** Equal when the spaces are. */
  friend bool operator==(const BoundSpace&, const BoundSpace&) = default;

 private:
  template <typename Difference, std::size_t... Slot>
  auto boundDifference(std::index_sequence<Slot...>) const {
    return BoundSpace<Difference, internal::differenceSlotQuantity<S, Difference, int(Slot), Qs...>()...>{
        differenceSpace(unbound())};
  }
};

/** \ingroup Spaces_Module
 * \brief A binding of the unit \a U to the block of type \a Tag, an argument of In; written bind<Tag>(unit).
 */
template <typename Tag, auto U>
struct TagBinding {
  using TagType = Tag;             ///< the block bound
  static constexpr auto unit = U;  ///< its unit
};

/** \ingroup Spaces_Module
 * \returns a binding of \a unit to the block of type \a Tag, an argument of In:
 * <tt>In<State, bind<Position>(si::metre), bind<Velocity>(si::metre / si::second)></tt>. A unit is an empty type
 * whose value its type determines, as mp-units' are, so the binding keeps it as a template argument.
 */
template <typename Tag, typename U>
  requires std::is_empty_v<U> && std::default_initializable<U>
consteval TagBinding<Tag, U{}> bind(U) {
  return {};
}

namespace internal {

template <typename T>
struct IsTagBindingImpl : std::false_type {};
template <typename Tag, auto U>
struct IsTagBindingImpl<TagBinding<Tag, U>> : std::true_type {};

/** \internal A binding by tag. */
template <typename T>
concept IsTagBinding = IsTagBindingImpl<std::remove_cvref_t<T>>::value;

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_BOUND_H
