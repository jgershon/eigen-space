// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_UNITS_BACKEND_H
#define EIGEN_UNITS_BACKEND_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal An mp-units quantity spec: what a tag's leaves measure before a unit is bound. */
template <typename T>
concept MpQuantitySpec = mp_units::QuantitySpec<std::remove_cvref_t<T>>;

/** \internal An mp-units reference, which includes a unit with an associated quantity. */
template <typename T>
concept MpReference = mp_units::Reference<std::remove_cvref_t<T>>;

/** \internal An mp-units unit given alone, whose kind is the unit's, as in <tt>quantity<si::metre></tt>. */
template <typename T>
concept MpUnitAlone = mp_units::Unit<std::remove_cvref_t<T>>;

/** \internal The mp-units backend of the quantity helpers in Core/Spaces/Unit.h (design/decisions.md
 * [mp-units-dependency]): Eigen's dimensionless is mp-units' dimensionless quantity in the unit one, and every other
 * quantity is mp-units' own.
 */
template <typename Q>
struct units_backend {
  /** The mp-units form of \a q. */
  template <typename A>
  static consteval auto toMp(A q) {
    if constexpr (IsEigenDimensionless<A>) {
      return mp_units::one;
    } else {
      static_assert(MpQuantitySpec<A> || MpReference<A>,
                    "a leaf measures an mp-units quantity spec, unit, or reference, or Eigen's dimensionless");
      return q;
    }
  }

  /** What \a q measures, without its unit. */
  template <typename A>
  static consteval auto specOf(A q) {
    using M = decltype(toMp(q));
    if constexpr (MpQuantitySpec<M>) {
      return M{};
    } else {
      return mp_units::get_quantity_spec(M{});
    }
  }

  template <typename A>
  static consteval bool isBound(A q) {
    return MpReference<decltype(toMp(q))>;
  }

  template <typename A>
  static consteval auto unitOf(A q) {
    static_assert(isBound(A{}), "the quantity has no unit; bind one with In");
    return mp_units::get_unit(toMp(q));
  }

  /** Whether \a q is of the dimensionless kind, in a unit of any magnitude. */
  template <typename A>
  static consteval bool isDimensionlessKind(A q) {
    return std::is_same_v<decltype(mp_units::get_kind(specOf(q))),
                          decltype(mp_units::get_kind(mp_units::dimensionless))>;
  }

  /** Whether \a q has a unit of dimension one and magnitude one, as the radian and the unit one have: a number of
   * radians, whatever its kind. A degree is not one.
   */
  template <typename A>
  static consteval bool inUnitOne(A q) {
    if constexpr (!isBound(A{})) {
      return false;
    } else {
      return equivalent(unitOf(q), mp_units::one);
    }
  }

  /** Whether \a q is a plain number: of the kind dimensionless, not merely of dimension one, so that an angle in
   * radians is not, as mp-units does not add one to a number; and in a unit of magnitude one, so that a percentage
   * is not either.
   */
  template <typename A>
  static consteval bool isDimensionless(A q) {
    if constexpr (!isDimensionlessKind(A{})) {
      return false;
    } else if constexpr (isBound(A{})) {
      return equivalent(unitOf(q), mp_units::one);
    } else {
      return true;
    }
  }

  /** Whether \a a and \a b have a common spec, mp-units' condition for adding them. */
  template <typename A, typename B>
  static consteval bool haveCommonSpec(A a, B b) {
    using SA = decltype(specOf(a));
    using SB = decltype(specOf(b));
    return requires { mp_units::get_common_quantity_spec(SA{}, SB{}); };
  }

  template <typename A, typename B>
  static consteval bool same(A a, B b) {
    if constexpr (std::is_same_v<decltype(toMp(a)), decltype(toMp(b))>) {
      return true;
    } else if constexpr (isBound(A{}) != isBound(B{})) {
      return false;
    } else if constexpr (!std::is_same_v<decltype(specOf(a)), decltype(specOf(b))>) {
      return false;
    } else if constexpr (isBound(A{})) {
      return equivalent(unitOf(a), unitOf(b));
    } else {
      return true;
    }
  }

  template <typename A, typename B>
  static consteval bool compatible(A a, B b) {
    return isBound(a) && isBound(b) && haveCommonSpec(a, b);
  }

  /** Whether \a a and \a b are in equivalent units, so that nothing is converted between them. */
  template <typename A, typename B>
  static consteval bool sameUnit(A a, B b) {
    if constexpr (!isBound(A{}) || !isBound(B{})) {
      return false;
    } else {
      return equivalent(unitOf(a), unitOf(b));
    }
  }

  /** Whether a value of \a from converts into \a to implicitly, as mp-units converts a quantity. */
  template <typename A, typename B>
  static consteval bool assignable(A from, B to) {
    if constexpr (!compatible(A{}, B{})) {
      return false;
    } else {
      return mp_units::implicitly_convertible(specOf(from), specOf(to));
    }
  }

  /** The quantity both \a a and \a b convert into: the common spec, in the common unit. */
  template <typename A, typename B>
  static consteval auto common(A a, B b) {
    static_assert(compatible(A{}, B{}), "the quantities have no common quantity to add in");
    constexpr auto unit = mp_units::get_common_unit(unitOf(A{}), unitOf(B{}));
    using MA = decltype(toMp(a));
    using MB = decltype(toMp(b));
    if constexpr (MpUnitAlone<MA> && MpUnitAlone<MB>) {
      return unit;
    } else {
      return mp_units::get_common_quantity_spec(specOf(a), specOf(b))[unit];
    }
  }

  /** The number a value in the unit of \a from is multiplied by to be in the unit of \a to. */
  template <typename A, typename B>
  static consteval long double factor(A from, B to) {
    return (1.0L * unitOf(from)).numerical_value_in(unitOf(to));
  }

  template <typename A>
  static consteval auto inverse(A q) {
    using M = decltype(toMp(q));
    if constexpr (MpQuantitySpec<M>) {
      return mp_units::dimensionless / M{};
    } else {
      return mp_units::one / M{};
    }
  }

  template <typename A, typename B>
  static consteval auto product(A a, B b) {
    return toMp(a) * toMp(b);
  }

  /** Whether a coordinate measuring \a q holds a \a Scalar of the right character [complex-scalars]: complex for a
   * complex spec, real for a real one, as mp-units' RepresentationOf checks; anything for a unit given alone, as for
   * mp-units' kind_of references, and for a vector or tensor spec, whose character is recorded and not checked. */
  template <typename Scalar, typename A>
  static consteval bool scalarFits(A q) {
    using M = decltype(toMp(q));
    if constexpr (IsEigenDimensionless<A> || MpUnitAlone<M>) {
      return true;
    } else {
      constexpr auto character = decltype(specOf(q))::character;
      if constexpr (character == mp_units::quantity_character::complex_scalar) {
        return NumTraits<Scalar>::IsComplex;
      } else if constexpr (character == mp_units::quantity_character::real_scalar) {
        return !NumTraits<Scalar>::IsComplex;
      } else {
        return true;
      }
    }
  }

  /** Whether \a unit can be bound to \a spec. */
  template <typename A, typename U>
  static consteval bool fits(A spec, U unit) {
    using M = decltype(toMp(spec));
    return MpQuantitySpec<M> && requires { M{}[U{}]; };
  }

  /** \a spec measured in \a unit. */
  template <typename A, typename U>
  static consteval auto bind(A spec, U unit) {
    static_assert(fits(A{}, U{}), "the unit does not measure the leaf's quantity");
    return toMp(spec)[unit];
  }
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_UNITS_BACKEND_H
