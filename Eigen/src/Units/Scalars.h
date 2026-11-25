// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_UNITS_SCALARS_H
#define EIGEN_UNITS_SCALARS_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal \a m scaled by the mp-units quantity \a q: the coefficients times its number, the row space times what it
 * measures (design/decisions.md [heterogeneous-results]), so that <tt>dt * v</tt>, with dt in seconds and v in metres
 * per second, is in metres, untagged, since its dimension changed. */
template <auto R, typename Derived>
auto scaleByQuantity(typename Derived::Scalar number, const MatrixBase<Derived>& m) {
  using Row = typename ScaledSpaceOf<typename spaces_of<Derived>::RowSpace, R>::type;
  using Col = typename spaces_of<Derived>::ColSpace;
  return relabelSpaces<Row, Col>(number * m.derived());
}

}  // namespace internal

/** \returns \a m scaled by the mp-units quantity \a q, in its spaces times what \a q measures: <tt>dt * v</tt>. */
template <auto R, typename Rep, typename Derived>
auto operator*(const mp_units::quantity<R, Rep>& q, const MatrixBase<Derived>& m) {
  return internal::scaleByQuantity<R>(typename Derived::Scalar(q.numerical_value_in(q.unit)), m);
}

/** \returns \a m scaled by the mp-units quantity \a q: <tt>v * dt</tt>. */
template <typename Derived, auto R, typename Rep>
auto operator*(const MatrixBase<Derived>& m, const mp_units::quantity<R, Rep>& q) {
  return internal::scaleByQuantity<R>(typename Derived::Scalar(q.numerical_value_in(q.unit)), m);
}

/** \returns \a m divided by the mp-units quantity \a q: <tt>x / dt</tt>. */
template <typename Derived, auto R, typename Rep>
auto operator/(const MatrixBase<Derived>& m, const mp_units::quantity<R, Rep>& q) {
  return internal::scaleByQuantity<mp_units::one / R>(
      typename Derived::Scalar(1) / typename Derived::Scalar(q.numerical_value_in(q.unit)), m);
}

}  // namespace Eigen

#endif  // EIGEN_UNITS_SCALARS_H
