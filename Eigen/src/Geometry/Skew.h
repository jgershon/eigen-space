// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_GEOMETRY_SKEW_H
#define EIGEN_GEOMETRY_SKEW_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \geometry_module \ingroup Geometry_Module
 * \returns the skew-symmetric matrix \f$ [v]_\times \f$ of the 3-vector \a v, with \f$ [v]_\times w = v \times w \f$:
 * \f[ [v]_\times = \begin{pmatrix} 0 & -v_3 & v_2 \\ v_3 & 0 & -v_1 \\ -v_2 & v_1 & 0 \end{pmatrix}. \f]
 * Its rows are in \a v's row space and its columns dual to \a ColSpace, the space of the vectors \f$ w \f$ it is
 * applied to, so that \f$ [v]_\times w \f$ is in \a v's units: the derivative of a rotated vector in the rotation's
 * step is <tt>skew<DifferenceSpace<Attitude>>(v)</tt>, in the units of \a v per radian. mp-units converts a product
 * of radians and metres to metres only explicitly; naming the column space is that explicit step
 * (design/decisions.md [lie-groups]). For a plain \a v and the default \a ColSpace it is a plain 3 by 3 matrix.
 */
template <typename ColSpace = Linear<3>, typename Derived>
  requires MatrixExpression<Derived>
auto skew(const Derived& v) {
  static_assert(Derived::SizeAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "skew() takes a 3-vector");
  static_assert(valueDimensionAtCompileTime<ColSpace> == 3, "skew() acts on vectors of three coordinates");
  using Scalar = typename Derived::Scalar;
  using RowSpace = typename internal::spaces_of<Derived>::RowSpace;
  const typename Derived::PlainObject evaluated = v;
  const Map<const Matrix<Scalar, 3, 1>> c(evaluated.data());
  QMatrix<RowSpace, Dual<ColSpace>, Scalar> result;
  Map<Matrix<Scalar, 3, 3>> m(result.data());
  m << Scalar(0), -c(2), c(1), c(2), Scalar(0), -c(0), -c(1), c(0), Scalar(0);
  return result;
}

/** \geometry_module \ingroup Geometry_Module
 * \returns \f$ [\phi]_\times \f$ of the rotation vector \a phi as a map from \a ColSpace to \a RowSpace: the matrix
 * that turns a vector of \a ColSpace by the small rotation \a phi, rows in \a RowSpace and columns dual to \a ColSpace.
 * \a phi must measure radians (or be plain), which this form drops, as \f$ \phi \times v \f$ is in \a v's units; and
 * \a RowSpace and \a ColSpace must measure the same, so that the matrix is a rotation's generator on them:
 * <tt>skew<VelocitySI, VelocitySI>(omega * dt)</tt> in an inertial filter's error dynamics.
 */
template <typename RowSpace, typename ColSpace, typename Derived>
  requires MatrixExpression<Derived>
auto skew(const Derived& phi) {
  static_assert(Derived::SizeAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "skew() takes a 3-vector");
  static_assert(valueDimensionAtCompileTime<RowSpace> == 3 && valueDimensionAtCompileTime<ColSpace> == 3,
                "skew() maps vectors of three coordinates");
  using PhiSpace = typename internal::spaces_of<Derived>::RowSpace;
  static_assert(
      internal::hasUniformQuantity<PhiSpace>() && internal::inRadians(internal::uniformQuantity<PhiSpace>()),
      "skew<RowSpace, ColSpace>() takes a rotation vector in radians; for another vector v, skew<ColSpace>(v) "
      "has rows in v's units");
  static_assert(internal::spaces_alike<RowSpace, ColSpace>::value,
                "skew<RowSpace, ColSpace>() is a rotation's generator, a map between two spaces that measure the same");
  using Scalar = typename Derived::Scalar;
  const typename Derived::PlainObject evaluated = phi;
  const Map<const Matrix<Scalar, 3, 1>> c(evaluated.data());
  QMatrix<RowSpace, Dual<ColSpace>, Scalar> result;
  Map<Matrix<Scalar, 3, 3>> m(result.data());
  m << Scalar(0), -c(2), c(1), c(2), Scalar(0), -c(0), -c(1), c(0), Scalar(0);
  return result;
}

}  // namespace Eigen

#endif  // EIGEN_GEOMETRY_SKEW_H
