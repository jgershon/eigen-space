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
 * \brief The unit of a coordinate that has none: the identity of every unit algebra.
 *
 * Eigen defines no unit system of its own (design/decisions.md [mandatory-units]). A leaf space carries a unit from
 * any library whose units are structural values closed under product and quotient, such as mp-units, and this is the
 * one unit Eigen supplies: the default of every leaf, and what a product of units that cancel compares equal to. A
 * foreign unit absorbs it on either side of a product or quotient, and compares equal to it when the unit is the
 * dimensionless one of its own system.
 */
struct Dimensionless {
  friend constexpr Dimensionless operator*(Dimensionless, Dimensionless) { return {}; }
  friend constexpr Dimensionless operator/(Dimensionless, Dimensionless) { return {}; }
  friend constexpr bool operator==(Dimensionless, Dimensionless) { return true; }
};

/** \ingroup Spaces_Module
 * The dimensionless unit, the default unit of a leaf space.
 */
inline constexpr Dimensionless dimensionless{};

/** \ingroup Spaces_Module
 * \brief A type whose values are units: closed under product and quotient, with constant-expression equality, and not
 * a number.
 *
 * Linear and Periodic take a value of such a type as their first template argument, so it must also be structural.
 * mp-units' unit objects satisfy it; so does Dimensionless. Arithmetic types are excluded, so that
 * <tt>QLinear<3></tt> is an error rather than a leaf whose unit is the integer three.
 */
template <typename U>
concept Unit = !std::is_arithmetic_v<std::remove_cvref_t<U>> && requires(const U& a, const U& b) {
  a * b;
  a / b;
  { a == b } -> std::convertible_to<bool>;
};

namespace internal {

// The operators between a foreign unit and Dimensionless below are templates, so they are candidates whenever
// Unit<U> is evaluated for a U of the foreign system. Their constraints check the Dimensionless operand first, so
// that Unit<U> is never re-entered while it is being evaluated, which MSVC reports as a constraint satisfied as
// false; the Dimensionless operand is a deduced parameter for the same reason.
template <typename D, typename U>
concept ForeignUnitAndDimensionless =
    std::same_as<D, Dimensionless> && !std::same_as<std::remove_cvref_t<U>, Dimensionless> && Unit<U>;

}  // namespace internal

/** A foreign unit absorbs the dimensionless unit on either side of a product or quotient. */
template <typename D, typename U>
  requires internal::ForeignUnitAndDimensionless<D, U>
constexpr U operator*(const U& u, D) {
  return u;
}
template <typename D, typename U>
  requires internal::ForeignUnitAndDimensionless<D, U>
constexpr U operator*(D, const U& u) {
  return u;
}
template <typename D, typename U>
  requires internal::ForeignUnitAndDimensionless<D, U>
constexpr U operator/(const U& u, D) {
  return u;
}
/** The inverse of a foreign unit: the dimensionless unit of its own system, <tt>u / u</tt>, divided by it. */
template <typename D, typename U>
  requires internal::ForeignUnitAndDimensionless<D, U>
constexpr auto operator/(D, const U& u) {
  return (u / u) / u;
}
/** A foreign unit equals the dimensionless unit when it is the dimensionless unit of its own system. */
template <typename D, typename U>
  requires internal::ForeignUnitAndDimensionless<D, U>
constexpr bool operator==(const U& u, D) {
  return u == u / u;
}

namespace internal {

/** \internal Whether \a u is dimensionless: the identity of its own unit algebra. */
template <Unit U>
constexpr bool isDimensionless(const U& u) {
  return u == dimensionless;
}

/** \internal Whether the units \a a and \a b are the same unit: equal where their types compare, and otherwise both
 * dimensionless, which is the one case where units of different systems can agree.
 */
template <Unit A, Unit B>
constexpr bool unitsEqual(const A& a, const B& b) {
  if constexpr (requires {
                  { a == b } -> std::convertible_to<bool>;
                }) {
    return a == b;
  } else {
    return isDimensionless(a) && isDimensionless(b);
  }
}

/** \internal The inverse of \a u: what a coordinate of the dual space is measured in. */
template <Unit U>
constexpr auto inverseUnit(const U& u) {
  return dimensionless / u;
}

/** \internal Whether a space type \a S declares its unit with a static unit() member, as the leaves do. */
template <typename S>
concept HasUnit = requires {
  { S::unit() } -> Unit;
};

/** \internal The unit of every coordinate of the single-block space \a S: its unit() if it declares one, and otherwise
 * dimensionless, which is what a space written from scratch gets.
 */
template <typename S>
constexpr auto leafUnit() {
  if constexpr (HasUnit<S>) {
    return S::unit();
  } else {
    return dimensionless;
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_UNIT_H
