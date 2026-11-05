// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The unit quaternion space of Geometry/UnitQuaternion.h (design/decisions.md [unit-quaternion]): its storage order,
// which is Eigen's (x, y, z, w); its body-frame plus() and minus(), their inverse relation, and the double cover; the
// exponential, logarithm, and right Jacobian; and map(), which views a block as a Quaternion without copying it, by
// itself, by tag path, and by repetition, writable or const. Expected values come from Eigen's own AngleAxis and
// quaternion product, an implementation independent of the space's.

#include "main.h"

#include <Eigen/Geometry>
#include <Eigen/Spaces>

#include <array>
#include <cmath>
#include <numbers>
#include <type_traits>

#include "unit_quaternion_helpers.h"

namespace {

struct Attitude : UnitQuaternion {};
struct Position : Linear<3> {};
struct Velocity : Linear<3> {};
using Body = Composite<Position, Attitude, Velocity>;

struct Vehicle : Composite<Position, Attitude> {};

struct Fleet : Repeated<Vehicle, 4> {
  using Repeated::Repeated;
};

struct Attitudes : Repeated<Attitude, 4> {
  using Repeated::Repeated;
};

static_assert(Space<UnitQuaternion>);
static_assert(Space<Body>);
static_assert(std::is_same_v<DifferenceSpace<UnitQuaternion>, Linear<3>>);
static_assert(!hasIdentityReset<UnitQuaternion>);
static_assert(valueDimensionAtCompileTime<Body> == 10 && valueDimensionAtCompileTime<DifferenceSpace<Body>> == 9);

const Vector3d axis = Vector3d{1.0, 2.0, -1.0}.normalized();
const Quaterniond turned{AngleAxisd{0.9, axis}};

// The steps the inverse tests take: zero, below the series threshold, moderate, large, and 0.04 rad
// short of π, where the logarithm is least well conditioned.
const std::array<Vector3d, 5> steps{Vector3d::Zero(), Vector3d{1e-7, -2e-7, 0.5e-7}, Vector3d{0.1, -0.2, 0.3},
                                    Vector3d{1.0, 0.5, -0.7}, (std::numbers::pi - 0.04) * Vector3d{0.0, 0.6, 0.8}};

// The coordinates are Eigen's coeffs(), (x, y, z, w): the exponential of a rotation vector is
// Eigen's AngleAxis quaternion coefficient for coefficient, and rotating through map() of the
// coordinates is Eigen's rotation matrix. 1e-15 is rounding in sin and cos.
void stores_eigens_coefficient_order() {
  const Vector4d q = UnitQuaternion::exponential(0.9 * axis).coeffs();
  VERIFY((q - turned.coeffs()).norm() < 1e-15);
  VERIFY(q(3) > 0.0);
  VERIFY_IS_APPROX(q(3), std::cos(0.45));
  const auto u = Vector3d{0.3, -1.0, 2.0};
  VERIFY((UnitQuaternion::map(q) * u - turned.toRotationMatrix() * u).norm() < 1e-15);
  VERIFY_IS_EQUAL(Quaterniond::FromCoeffsScalarFirst(q(3), q(0), q(1), q(2)).coeffs(), q);
}

// identity() and coefficientsOf() write a mean in the space's order without counting where w goes:
// the identity is w = 1 last, and a quaternion's coefficients are its coeffs().
void builds_values_in_its_own_order() {
  VERIFY_IS_EQUAL(UnitQuaternion::identity(), Vector4d(0.0, 0.0, 0.0, 1.0));
  VERIFY_IS_EQUAL(UnitQuaternion::identity<float>(), Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
  VERIFY_IS_EQUAL(UnitQuaternion::coefficientsOf(turned), turned.coeffs());
  const Vector4d identity = UnitQuaternion::identity();
  VERIFY_IS_EQUAL(UnitQuaternion::map(identity).coeffs(), Quaterniond::Identity().coeffs());
}

// normalize() scales to unit norm and picks w >= 0, so q and -q give one value; minus() agrees on
// either sign of either argument, since they are the same rotation.
void normalizes_onto_one_hemisphere() {
  Vector4d q = -3.0 * turned.coeffs();
  normalizeValues(UnitQuaternion{}, q);
  VERIFY((q - turned.coeffs()).norm() < 1e-15);

  const Vector4d p = Quaterniond{AngleAxisd{0.4, Vector3d::UnitZ()}}.coeffs();
  const Vector3d d = minus(UnitQuaternion{}, q, p);
  VERIFY((minus(UnitQuaternion{}, Vector4d{-q}, p) - d).norm() < 1e-15);
  VERIFY((minus(UnitQuaternion{}, q, Vector4d{-p}) - d).norm() < 1e-15);
}

// plus() applies a step in the body frame, x ⊗ Exp(d), which is Eigen's product with the AngleAxis
// of the step on the right; minus() inverts it at every step, near π included, where 1e-12 is the
// conditioning of atan2 there.
void plus_is_body_frame_and_minus_inverts_it() {
  for (const auto& d : steps) {
    const Vector4d moved = plus(UnitQuaternion{}, turned.coeffs(), d);
    Quaterniond step = Quaterniond::Identity();
    if (d.norm() > 0) {
      step = AngleAxisd{d.norm(), d.normalized()};
    }
    const Quaterniond expected = turned * step;
    VERIFY(std::abs(std::abs(expected.coeffs().dot(moved)) - 1.0) < 1e-15);
    VERIFY(std::abs(moved.norm() - 1.0) <= 1e-15);
    VERIFY(moved(3) >= 0.0);
    VERIFY((minus(UnitQuaternion{}, moved, turned.coeffs()) - d).norm() < 1e-12);
  }
}

// The logarithm takes the short way round: a rotation of 3π/2 about an axis comes back as π/2
// about the opposite one.
void logarithm_takes_the_short_way() {
  const auto far = Quaterniond{AngleAxisd{1.5 * std::numbers::pi, Vector3d::UnitX()}};
  const Vector3d d = UnitQuaternion::logarithm(far);
  VERIFY((d - Vector3d{-0.5 * std::numbers::pi, 0.0, 0.0}).norm() < 1e-15);
  VERIFY((UnitQuaternion::logarithm(UnitQuaternion::exponential(d)) - d).norm() < 1e-15);
}

// The space's reset Jacobian is its closed-form right Jacobian, which the reset computed by
// automatic differentiation of the same plus() and minus() equals to rounding [covariance-reset], and which is
// also the derivative of plus() in its step: the change of x ⊞ (d + ε) seen from x ⊞ d, by central
// differences, to 1e-9 for a step h of 1e-6.
void reset_jacobian_is_the_right_jacobian() {
  for (const auto& d : steps) {
    const Matrix3d closed = resetJacobian(UnitQuaternion{}, turned.coeffs(), d);
    VERIFY_IS_EQUAL(closed, UnitQuaternion::rightJacobian(d));
    const Matrix3d automatic = resetJacobian(spacetest::AutoDiffQuaternion{}, turned.coeffs(), d);
    VERIFY((automatic - closed).norm() < 1e-12);
  }
  const Vector3d d = steps[3];
  const Vector4d moved = plus(UnitQuaternion{}, turned.coeffs(), d);
  const double h = 1e-6;
  for (int j = 0; j < 3; ++j) {
    const Vector3d e = h * Vector3d::Unit(j);
    const Vector4d ahead = plus(UnitQuaternion{}, turned.coeffs(), Vector3d{d + e});
    const Vector4d behind = plus(UnitQuaternion{}, turned.coeffs(), Vector3d{d - e});
    const Vector3d column = (minus(UnitQuaternion{}, ahead, moved) - minus(UnitQuaternion{}, behind, moved)) / (2 * h);
    VERIFY((column - UnitQuaternion::rightJacobian(d).col(j)).norm() < 1e-9);
  }
}

// map() of a vector, a const vector, and a column of a matrix is a map over the same memory:
// writable where the argument is, const where it is not, and writing through it writes the vector.
void map_views_the_block_without_copying() {
  Vector4d q = turned.coeffs();
  auto writable = UnitQuaternion::map(q);
  static_assert(std::is_same_v<decltype(writable), Map<Quaterniond>>);
  VERIFY_IS_EQUAL(writable.coeffs().data(), q.data());
  writable = Quaterniond::Identity();
  VERIFY_IS_EQUAL(q, Vector4d::UnitW());

  const Vector4d fixed = turned.coeffs();
  const auto view = UnitQuaternion::map(fixed);
  static_assert(std::is_same_v<std::remove_const_t<decltype(view)>, Map<const Quaterniond>>);
  VERIFY_IS_EQUAL(view.coeffs().data(), fixed.data());

  Matrix<double, 4, 3> states = Matrix<double, 4, 3>::Zero();
  UnitQuaternion::map(states.col(1)) = turned;
  VERIFY_IS_EQUAL(Vector4d{states.col(1)}, turned.coeffs());
  VERIFY_IS_EQUAL(states.col(0), Vector4d::Zero());

  const Matrix<double, 4, 3> constStates = states;
  static_assert(std::is_same_v<decltype(UnitQuaternion::map(constStates.col(1))), Map<const Quaterniond>>);
  VERIFY_IS_EQUAL(UnitQuaternion::map(constStates.col(1)).coeffs(), turned.coeffs());
}

// map<Path>(space, x) finds the block by tag path in a whole state, after a position's three
// coordinates, and map<Path>(space, x, element) one repetition of a Repeated block after a
// position, the repeated block itself or the Attitude of a repeated Composite; each is a view at
// the block's own address.
void map_finds_the_block_by_tag_path() {
  auto x = Matrix<double, 10, 1>{};
  x << 1.0, 2.0, 3.0, turned.coeffs(), 0.1, 0.2, 0.3;
  const auto attitude = UnitQuaternion::map<Attitude>(Body{}, x);
  VERIFY_IS_EQUAL(attitude.coeffs().data(), x.data() + 3);
  VERIFY(attitude.isApprox(turned));
  UnitQuaternion::map<Attitude>(Body{}, x) = Quaterniond::Identity();
  VERIFY_IS_EQUAL(Vector4d{x.segment<4>(3)}, Vector4d::UnitW());

  using Formation = Composite<Position, Attitudes>;
  const auto formation = Formation{Position{}, Attitudes{Attitude{}, 3}};
  const Matrix<double, 15, 1> values = Matrix<double, 15, 1>::Zero();
  VERIFY_IS_EQUAL(UnitQuaternion::map<Attitudes>(formation, values, 2).coeffs().data(), values.data() + 11);

  using Convoy = Composite<Position, Fleet>;
  const auto convoy = Convoy{Position{}, Fleet{Vehicle{}, 3}};
  Matrix<double, 24, 1> vehicles = Matrix<double, 24, 1>::Zero();
  const auto second = UnitQuaternion::map<TagPath<Fleet, Attitude>>(convoy, vehicles, 1);
  VERIFY_IS_EQUAL(second.coeffs().data(), vehicles.data() + 13);
}

}  // namespace

EIGEN_DECLARE_TEST(geo_unit_quaternion) {
  CALL_SUBTEST_1(stores_eigens_coefficient_order());
  CALL_SUBTEST_1(builds_values_in_its_own_order());
  CALL_SUBTEST_1(normalizes_onto_one_hemisphere());
  CALL_SUBTEST_1(plus_is_body_frame_and_minus_inverts_it());
  CALL_SUBTEST_1(logarithm_takes_the_short_way());
  CALL_SUBTEST_2(reset_jacobian_is_the_right_jacobian());
  CALL_SUBTEST_2(map_views_the_block_without_copying());
  CALL_SUBTEST_2(map_finds_the_block_by_tag_path());
}
