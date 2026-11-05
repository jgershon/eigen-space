// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The covariance reset of Eigen/Spaces (design/decisions.md [covariance-reset]): hasIdentityReset<>, resetJacobian(),
// and resetCovariance(). The unit quaternion is the space with a reset, by automatic differentiation
// (AutoDiffQuaternion, from test/unit_quaternion_helpers.h) and in closed form (UnitQuaternion).

#include "main.h"

#include <Eigen/Spaces>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <numbers>

#include "unit_quaternion_helpers.h"

namespace {

using spacetest::AutoDiffQuaternion;
using spacetest::exponential;
using spacetest::QuaternionVector;
using spacetest::rightJacobian;

// A positive gain, x ⊞ δ = x e^δ and a ⊟ b = log(a / b), whose reset Jacobian is exactly one and
// which says so.
struct Gain {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr bool identityReset = true;

  static constexpr Index valueDimension() { return 1; }

  Linear<1> difference() const { return {}; }

  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    writable(out) = x.array() * d.array().exp();
  }

  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    writable(out) = (a.array() / b.array()).log();
  }

  friend bool operator==(const Gain&, const Gain&) = default;
};

// The same gain without the declaration, so its reset is differentiated.
struct UndeclaredGain : Gain {
  static constexpr bool identityReset = false;
  friend bool operator==(const UndeclaredGain&, const UndeclaredGain&) = default;
};

struct Position : Linear<3> {};
struct Attitude : AutoDiffQuaternion {};
struct ClosedFormAttitude : UnitQuaternion {};
struct Velocity : Linear<3> {};
struct Heading : RadiansPi<> {};
struct Range : Linear<1> {};

using Body = Composite<Position, Attitude, Velocity>;
using ClosedFormBody = Composite<Position, ClosedFormAttitude, Velocity>;

struct Attitudes : Repeated<Attitude, 3> {
  using Repeated::Repeated;
};

using Formation = Composite<Position, Attitudes>;

static_assert(hasIdentityReset<Linear<3>>);
static_assert(hasIdentityReset<Heading>);
static_assert(hasIdentityReset<Range>);
static_assert(hasIdentityReset<Composite<Position, Heading, Range>>);
static_assert(hasIdentityReset<Repeated<Composite<Heading, Range>>>);
static_assert(hasIdentityReset<Gain>);
static_assert(hasIdentityReset<Composite<Position, Gain>>);
static_assert(!hasIdentityReset<UndeclaredGain>);
static_assert(!hasIdentityReset<AutoDiffQuaternion>);
static_assert(!hasIdentityReset<UnitQuaternion>);
static_assert(!hasIdentityReset<Body>);
static_assert(!hasIdentityReset<Attitudes>);
static_assert(!hasIdentityReset<Formation>);

const QuaternionVector<double> turned = exponential(Vector3d{0.3, -0.2, 0.5});

// A symmetric positive definite matrix of size N with every entry nonzero, from a fixed square.
template <int N>
Matrix<double, N, N> spreadCov() {
  auto a = Matrix<double, N, N>{};
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      a(i, j) = std::sin(1.0 + 3.0 * i + 7.0 * j);
    }
  }
  // Symmetrized exactly: the product alone is symmetric only to rounding, and the tests check that the reset keeps
  // an exactly symmetric input exactly symmetric.
  const Matrix<double, N, N> m = a * a.transpose() + 0.5 * Matrix<double, N, N>::Identity();
  return 0.5 * (m + m.transpose());
}

// The mean of a Body at turned, with the position and velocity given.
Matrix<double, 10, 1> bodyMean() {
  auto mean = Matrix<double, 10, 1>{};
  mean << 1.0, -2.0, 0.5, turned, 0.3, 0.1, -0.2;
  return mean;
}

// The automatically differentiated reset Jacobian of the quaternion equals the closed-form right
// Jacobian J_r(d) of Solà, Deray, and Atchuthan (2018), eq. (143), at a zero step, a small one, a
// large one, and one 0.04 rad short of π, from two starting attitudes; the closed form the space
// writes is used as it is. 1e-12 is rounding in the trigonometric functions near π.
void quaternion_jacobian_is_the_right_jacobian() {
  const Vector3d axis = Vector3d{1.0, 2.0, -1.0}.normalized();
  const auto steps = std::array<Vector3d, 5>{Vector3d::Zero(), Vector3d{1e-7, -2e-7, 0.5e-7}, Vector3d{0.1, -0.2, 0.3},
                                             Vector3d{1.0, 0.5, -0.7}, (std::numbers::pi - 0.04) * axis};
  const auto starts = std::array<QuaternionVector<double>, 2>{QuaternionVector<double>::UnitW(), turned};
  for (const auto& x : starts) {
    for (const auto& d : steps) {
      const Matrix3d expected = rightJacobian(d);
      const Matrix3d automatic = resetJacobian(AutoDiffQuaternion{}, x, d);
      VERIFY((automatic - expected).norm() < 1e-12);
      const Matrix3d closed = resetJacobian(UnitQuaternion{}, x, d);
      VERIFY_IS_EQUAL(closed, expected);
    }
  }
  // At a zero step the reset is the identity, as it must be: nothing moved.
  VERIFY((resetJacobian(AutoDiffQuaternion{}, turned, Vector3d::Zero()) - Matrix3d::Identity()).norm() < 1e-15);
}

// The Jacobian is the derivative [covariance-reset] defines: a small error ε at x, carried to x ⊞ d, is
// (x ⊞ (d + ε)) ⊟ (x ⊞ d) ≈ G ε. Checked by central differences of plus() and minus() themselves,
// whose truncation error at a step of 1e-5 is about 1e-10.
void jacobian_carries_an_error_to_the_moved_value() {
  const auto d = Vector3d{0.4, -0.9, 0.6};
  const QuaternionVector<double> moved = plus(AutoDiffQuaternion{}, turned, d);
  const Matrix3d g = resetJacobian(AutoDiffQuaternion{}, turned, d);
  const double h = 1e-5;
  for (int j = 0; j < 3; ++j) {
    const Vector3d e = h * Vector3d::Unit(j);
    const QuaternionVector<double> ahead = plus(AutoDiffQuaternion{}, turned, Vector3d{d + e});
    const QuaternionVector<double> behind = plus(AutoDiffQuaternion{}, turned, Vector3d{d - e});
    const Vector3d column =
        (minus(AutoDiffQuaternion{}, ahead, moved) - minus(AutoDiffQuaternion{}, behind, moved)) / (2 * h);
    VERIFY((column - g.col(j)).norm() < 1e-9);
  }
}

// On position, attitude, and velocity, resetJacobian() is the identity but for J_r on the attitude,
// resetCovariance() equals the dense G P G^T to rounding, leaves the position and velocity rows and
// columns where they meet bit for bit as they were, and returns an exactly symmetric matrix; the
// closed form gives the same covariance to rounding.
void covariance_matches_the_dense_congruence() {
  const auto space = Body{};
  const Matrix<double, 10, 1> x = bodyMean();
  auto d = Matrix<double, 9, 1>{};
  d << 0.2, 0.1, -0.3, 0.5, -0.4, 0.8, 0.05, 0.0, -0.1;
  const Matrix<double, 9, 9> p = spreadCov<9>();

  Matrix<double, 9, 9> expectedG = Matrix<double, 9, 9>::Identity();
  expectedG.block<3, 3>(3, 3) = rightJacobian(Vector3d{d.segment<3>(3)});
  const Matrix<double, 9, 9> g = resetJacobian(space, x, d);
  VERIFY((g - expectedG).norm() < 1e-12);

  Matrix<double, 9, 9> reset = p;
  resetCovariance(space, x, d, reset);
  const Matrix<double, 9, 9> dense = g * p * g.transpose();
  VERIFY((reset - dense).norm() < 1e-13 * p.norm());
  VERIFY_IS_EQUAL(reset, reset.transpose());
  const auto plain = std::array<int, 6>{0, 1, 2, 6, 7, 8};
  for (int i : plain) {
    for (int j : plain) {
      VERIFY_IS_EQUAL(reset(i, j), p(i, j));
    }
  }

  Matrix<double, 9, 9> closed = p;
  resetCovariance(ClosedFormBody{}, x, d, closed);
  VERIFY((closed - reset).norm() < 1e-12 * p.norm());
}

// In a Repeated block of quaternions after a position, each repetition is reset by its own J_r at
// its own value and difference offsets, which differ (four value coordinates, three difference
// ones), and the result equals the dense congruence by blockdiag(I, J_r(d_1), J_r(d_2), J_r(d_3)).
// minus() of the moved formation and the old one is the step, each repetition by its own minus().
void repeated_blocks_reset_each_repetition() {
  const auto space = Formation{Position{}, Attitudes{Attitude{}, 3}};
  auto x = Matrix<double, 15, 1>{};
  x << 1.0, 2.0, 3.0, turned, QuaternionVector<double>::UnitW(), exponential(Vector3d{-0.4, 0.1, 0.2});
  auto d = Matrix<double, 12, 1>{};
  d << 0.3, 0.3, 0.3, 0.2, -0.1, 0.4, 0.0, 0.0, 0.0, -0.6, 0.5, 0.2;
  const MatrixXd p = spreadCov<12>();

  Matrix<double, 12, 12> g = Matrix<double, 12, 12>::Identity();
  for (int k = 0; k < 3; ++k) {
    g.block<3, 3>(3 + 3 * k, 3 + 3 * k) = rightJacobian(Vector3d{d.segment<3>(3 + 3 * k)});
  }
  VERIFY((resetJacobian(space, x, d) - g).norm() < 1e-12);
  MatrixXd reset = p;
  resetCovariance(space, x, d, reset);
  VERIFY((reset - g * p * g.transpose()).norm() < 1e-13 * p.norm());
  VERIFY_IS_EQUAL(reset, reset.transpose());

  const Matrix<double, 15, 1> moved = plus(space, x, d);
  VERIFY((minus(space, moved, x) - d).norm() < 1e-12);
}

// A declared identity reset leaves a covariance bit for bit as it was, and the same gain without the
// declaration differentiates a Jacobian of one, so its reset changes the covariance only by rounding.
void identity_resets_leave_the_covariance_as_it_was() {
  const Vector4d x{0.5, -1.0, 0.3, 2.0};
  const Vector4d d{0.1, 0.2, -0.3, 0.4};
  const Matrix4d cov = spreadCov<4>();

  Matrix4d declared = cov;
  resetCovariance(Composite<Linear<3>, Gain>{}, x, d, declared);
  VERIFY_IS_EQUAL(declared, cov);

  Matrix4d blocks = cov;
  resetCovariance(Composite<Linear<2>, RadiansPi<>, Linear<1>>{}, x, d, blocks);
  VERIFY_IS_EQUAL(blocks, cov);

  Matrix4d undeclared = cov;
  resetCovariance(Composite<Linear<3>, UndeclaredGain>{}, x, d, undeclared);
  VERIFY((undeclared - cov).norm() < 1e-14 * cov.norm());
  const Matrix4d g = resetJacobian(Composite<Linear<3>, UndeclaredGain>{}, x, d);
  VERIFY((g - Matrix4d::Identity()).norm() < 1e-15);
}

}  // namespace

EIGEN_DECLARE_TEST(spaces_reset) {
  CALL_SUBTEST_1(quaternion_jacobian_is_the_right_jacobian());
  CALL_SUBTEST_1(jacobian_carries_an_error_to_the_moved_value());
  CALL_SUBTEST_2(covariance_matches_the_dense_congruence());
  CALL_SUBTEST_2(repeated_blocks_reset_each_repetition());
  CALL_SUBTEST_2(identity_resets_leave_the_covariance_as_it_was());
}
