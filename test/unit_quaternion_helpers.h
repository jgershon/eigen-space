// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Unit quaternion helpers for the space tests, over Eigen::UnitQuaternion (design/decisions.md [unit-quaternion]):
// coefficient vectors (x, y, z, w) with the product, conjugate, rotation, exponential, logarithm, and right Jacobian
// of SO(3), and AutoDiffQuaternion, the same space without its closed-form reset Jacobian, so that the reset comes
// from automatic differentiation of plus() and minus(). The two are the two paths of [covariance-reset].

#ifndef EIGEN_TEST_UNIT_QUATERNION_HELPERS_H
#define EIGEN_TEST_UNIT_QUATERNION_HELPERS_H

#include <Eigen/Geometry>

namespace spacetest {

template <typename T>
using QuaternionVector = Eigen::Matrix<T, 4, 1>;

// The product of the quaternions p and q, each (x, y, z, w).
template <typename P, typename Q>
QuaternionVector<typename P::Scalar> multiply(const P& p, const Q& q) {
  using T = typename P::Scalar;
  const QuaternionVector<T> a = p.template cast<T>();
  const QuaternionVector<T> b = q.template cast<T>();
  return (Eigen::Quaternion<T>{a} * Eigen::Quaternion<T>{b}).coeffs();
}

template <typename Q>
QuaternionVector<typename Q::Scalar> conjugate(const Q& q) {
  return QuaternionVector<typename Q::Scalar>{-q(0), -q(1), -q(2), q(3)};
}

// The unit quaternion of the rotation vector d.
template <typename D>
QuaternionVector<typename D::Scalar> exponential(const D& d) {
  return Eigen::UnitQuaternion::exponential(d).coeffs();
}

// The rotation vector of the unit quaternion q.
template <typename Q>
Eigen::Matrix<typename Q::Scalar, 3, 1> logarithm(const Q& q) {
  const QuaternionVector<typename Q::Scalar> fixed = q;
  return Eigen::UnitQuaternion::logarithm(Eigen::Quaternion<typename Q::Scalar>{fixed});
}

// v rotated by the unit quaternion q.
template <typename Q, typename V>
Eigen::Matrix<typename Q::Scalar, 3, 1> rotate(const Q& q, const V& v) {
  using T = typename Q::Scalar;
  const QuaternionVector<T> fixed = q;
  const Eigen::Matrix<T, 3, 1> u = v.template cast<T>();
  return Eigen::Quaternion<T>{fixed} * u;
}

// The right Jacobian of SO(3) at the rotation vector d.
template <typename T>
Eigen::Matrix<T, 3, 3> rightJacobian(const Eigen::Matrix<T, 3, 1>& d) {
  return Eigen::UnitQuaternion::rightJacobian(d);
}

// The library's unit quaternion without its closed-form reset Jacobian, so that the reset comes
// from automatic differentiation of plus() and minus().
struct AutoDiffQuaternion {
  static constexpr int FixedValueDimension = 4;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 4;

  static constexpr Eigen::Index valueDimension() { return 4; }

  template <typename Derived>
  void normalize(const Eigen::MatrixBase<Derived>& block) const {
    Eigen::UnitQuaternion{}.normalize(block);
  }

  Eigen::Linear<3> difference() const { return {}; }

  template <typename X, typename D, typename Out>
  void plus(const Eigen::MatrixBase<X>& x, const Eigen::MatrixBase<D>& d, const Eigen::MatrixBase<Out>& out) const {
    Eigen::UnitQuaternion{}.plus(x, d, out);
  }

  template <typename A, typename B, typename Out>
  void minus(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b, const Eigen::MatrixBase<Out>& out) const {
    Eigen::UnitQuaternion{}.minus(a, b, out);
  }

  friend bool operator==(const AutoDiffQuaternion&, const AutoDiffQuaternion&) = default;
};

}  // namespace spacetest

#endif  // EIGEN_TEST_UNIT_QUATERNION_HELPERS_H
