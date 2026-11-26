// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_UNIT_H
#define EIGEN_SPACES_UNIT_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup Spaces_Module
 * \brief What the coordinates of a plain space measure: nothing, in no unit.
 *
 * The first template argument of every leaf space is what its coordinates measure, an mp-units quantity spec, unit,
 * or reference (design/decisions.md [leaf-quantities]); the plain spaces, Linear and its kin, take this one, which
 * needs no units library. Eigen/Units maps it onto mp-units' dimensionless quantity in the unit one
 * [mp-units-dependency].
 */
struct Dimensionless {
  friend constexpr bool operator==(Dimensionless, Dimensionless) { return true; }
};

/** \ingroup Spaces_Module
 * The quantity of the plain spaces: dimensionless, in the unit one.
 */
inline constexpr Dimensionless dimensionless{};

namespace internal {

/** \internal The units library behind every quantity other than Eigen's dimensionless: declared here and defined by
 * Eigen/Units on mp-units (design/decisions.md [mp-units-dependency]), so that a space with units used without it
 * does not compile. \a Q only delays the instantiation to the point of use. Its members, each static and consteval,
 * take mp-units quantity specs, units, references, or Dimensionless:
 *  - isBound(q), isDimensionless(q), unitOf(q);
 *  - same(a, b): the same quantity in the same unit, so that nothing is converted;
 *  - compatible(a, b): quantities mp-units would add, in units that may differ;
 *  - assignable(from, to): what mp-units converts implicitly;
 *  - common(a, b), inverse(q), product(a, b), bind(spec, unit);
 *  - factor(from, to): the number a value in the unit of \a from is multiplied by to be in that of \a to.
 */
template <typename Q>
struct units_backend;

/** \internal Eigen's own dimensionless quantity. */
template <typename Q>
concept IsEigenDimensionless = std::same_as<std::remove_cvref_t<Q>, Dimensionless>;

/** \internal What a leaf's first argument may be: anything but a number, so that <tt>QLinear<3></tt> is an error
 * rather than a leaf measuring the integer three. Whether it is a quantity is mp-units' to say, at the point of use.
 */
template <typename Q>
concept QuantityArgument = !std::is_arithmetic_v<std::remove_cvref_t<Q>>;

/** \internal The backend for an operation on \a A and \a B: either one's, where the other is Eigen's dimensionless. */
template <typename A, typename B = A>
using units_backend_for = units_backend<std::conditional_t<IsEigenDimensionless<A>, B, A>>;

/** \internal Whether \a q has a unit: Eigen's dimensionless, an mp-units unit or reference, and not a bare spec. */
template <typename Q>
consteval bool isBound(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return true;
  } else {
    return units_backend<Q>::isBound(q);
  }
}

/** \internal Whether \a q is dimensionless in the unit one, so that a coordinate measuring it is a plain number. */
template <typename Q>
consteval bool isDimensionless(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return true;
  } else {
    return units_backend<Q>::isDimensionless(q);
  }
}

/** \internal Whether \a a and \a b are the same quantity in the same unit: nothing to convert between them. */
template <typename A, typename B>
consteval bool sameQuantity(const A& a, const B& b) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return true;
  } else {
    return units_backend_for<A, B>::same(a, b);
  }
}

/** \internal Whether a value of \a a and one of \a b add, after converting where their units differ. */
template <typename A, typename B>
consteval bool compatibleQuantities(const A& a, const B& b) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return true;
  } else {
    return units_backend_for<A, B>::compatible(a, b);
  }
}

/** \internal Whether \a a and \a b add with nothing converted: compatible quantities in equivalent units, such as
 * <tt>si::metre</tt> and <tt>isq::length[si::metre]</tt>. */
template <typename A, typename B>
consteval bool alikeQuantities(const A& a, const B& b) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return true;
  } else {
    return units_backend_for<A, B>::compatible(a, b) && units_backend_for<A, B>::sameUnit(a, b);
  }
}

/** \internal Whether a value of \a from converts into \a to implicitly, as mp-units converts a quantity. */
template <typename A, typename B>
consteval bool assignableQuantity(const A& from, const B& to) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return true;
  } else {
    return units_backend_for<A, B>::assignable(from, to);
  }
}

/** \internal The quantity a value of \a a and one of \a b add in: the common spec in the common unit. */
template <typename A, typename B>
consteval auto commonQuantity(const A& a, const B& b) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return a;
  } else {
    return units_backend_for<A, B>::common(a, b);
  }
}

/** \internal The number a coordinate measuring \a from is multiplied by to measure \a to. */
template <typename A, typename B>
consteval double conversionFactor(const A& from, const B& to) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return 1.0;
  } else {
    return double(units_backend_for<A, B>::factor(from, to));
  }
}

/** \internal The quantity of a product of a coordinate measuring \a a and one measuring \a b. */
template <typename A, typename B>
consteval auto quantityProduct(const A& a, const B& b) {
  if constexpr (IsEigenDimensionless<A> && IsEigenDimensionless<B>) {
    return a;
  } else {
    return units_backend_for<A, B>::product(a, b);
  }
}

/** \internal Whether \a q is of the dimensionless kind, in any unit: a ratio such as m/ft, which converts a value
 * without changing what it measures. */
template <typename Q>
consteval bool isDimensionlessKind(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return true;
  } else {
    return units_backend<Q>::isDimensionlessKind(q);
  }
}

/** \internal Whether a coordinate measuring \a q is a number of radians: Eigen's dimensionless, or a unit of dimension
 * one and magnitude one, such as the radian. The rotations' closed forms read their steps in radians.
 */
template <typename Q>
consteval bool inRadians(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return true;
  } else {
    return units_backend<Q>::inUnitOne(q);
  }
}

/** \internal Whether a coordinate measuring \a q holds a \a Scalar of the right character [complex-scalars]. */
template <typename Scalar, typename Q>
consteval bool scalarFitsQuantity(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return true;
  } else {
    return units_backend<Q>::template scalarFits<Scalar>(q);
  }
}

/** \internal The quantity of the dual coordinate: \a q inverted. */
template <typename Q>
consteval auto inverseQuantity(const Q& q) {
  if constexpr (IsEigenDimensionless<Q>) {
    return q;
  } else {
    return units_backend<Q>::inverse(q);
  }
}

/** \internal Whether a space type \a S declares what its coordinates measure with a static quantity(), as the leaves
 * do. */
template <typename S>
concept HasQuantity = requires { S::quantity(); };

/** \internal What every coordinate of the single-block space \a S measures: its quantity() if it declares one, and
 * otherwise dimensionless, which is what a space written from scratch gets.
 */
template <typename S>
consteval auto leafQuantity() {
  if constexpr (HasQuantity<S>) {
    return S::quantity();
  } else {
    return dimensionless;
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_UNIT_H
