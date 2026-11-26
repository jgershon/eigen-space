// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Lie groups as spaces, on plain coordinates (Spaces/Group.h, Spaces/LieGroup.h, Geometry/ExtendedPose.h), ported
// from EmrysFilter's groups-prototype: which spaces are groups; the members of Linear, Periodic, UnitQuaternion,
// Composite, and ExtendedPose through the free functions; and ExtendedPose of fixed size, of run-time size, and in
// the plane under a heading in degrees. checkGroup() checks each group's members against each other; the poses'
// expected values come from Eigen's 4 by 4 homogeneous matrices and its matrix exponential, independent of the space.

#include "main.h"

#include <Eigen/Geometry>
#include <Eigen/Spaces>
#include <unsupported/Eigen/MatrixFunctions>

#include <numbers>

#include "lie_group_helpers.h"

namespace {

using lie_group_test::checkGroup;

struct Attitude : UnitQuaternion {};
struct Velocity : Linear<3> {};
struct Position : Linear<3> {};
struct Foot : Linear<3> {};
struct Feet : Repeated<Foot> {
  using Repeated::Repeated;
};
struct Bias : Linear<3> {};
struct Heading : RadiansPi<> {};
struct Compass : Degrees360<> {};
struct Place : Linear<2> {};
struct Pose : ExtendedPose<Attitude, Position> {};
struct Navigation : ExtendedPose<Attitude, Velocity, Position> {};
struct Body : ExtendedPose<Attitude, Velocity, Position, Feet> {
  using ExtendedPose::ExtendedPose;
};
struct PlanarPose : ExtendedPose<Compass, Place> {};

// A space with a plus() of its own that is not a group: a unit direction would be one.
struct NotAGroup {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr Index valueDimension() { return 1; }
  Linear<1> difference() const { return {}; }
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    writable(out) = x + d;
  }
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    writable(out) = a - b;
  }
  [[maybe_unused]] friend bool operator==(const NotAGroup&, const NotAGroup&) = default;
};

static_assert(LieGroup<Linear<3>> && LieGroup<Heading> && LieGroup<UnitQuaternion> && LieGroup<Attitude>);
static_assert(LieGroup<Composite<Attitude, Bias>> && LieGroup<Repeated<Attitude>>);
static_assert(LieGroup<Pose> && LieGroup<Body> && LieGroup<Composite<Body, Bias>> && LieGroup<PlanarPose>);
static_assert(Space<NotAGroup> && !LieGroup<NotAGroup> && !LieGroup<Composite<Position, NotAGroup>>);
static_assert(Rotation<Attitude> && Rotation<Compass> && !Rotation<Position>);
static_assert(hasIdentityAdjoint<Composite<Position, Heading>> && !hasIdentityAdjoint<Attitude>);
static_assert(!hasIdentityAdjoint<Composite<Pose, Bias>>);
static_assert(Space<Body> && !hasIdentityReset<Body>);
static_assert(std::is_empty_v<Navigation>);
static_assert(valueDimensionAtCompileTime<Navigation> == 10);
static_assert(valueDimensionAtCompileTime<DifferenceSpace<Navigation>> == 9);
static_assert(valueDimensionAtCompileTime<Body> == Dynamic);
// A pose of a heading and a vector has blocks that add, but its own product: it is not a group under addition.
static_assert(!internal::isAdditiveGroup<PlanarPose>() && internal::isAdditiveGroup<Composite<Compass, Place>>());

constexpr double tolerance = 1e-12;

using Vector7d = Matrix<double, 7, 1>;
using Vector6d = Matrix<double, 6, 1>;

// The 4 by 4 homogeneous matrix of a pose value (x, y, z, w, p).
Matrix4d homogeneous(const Vector7d& x) {
  Matrix4d result = Matrix4d::Identity();
  result.topLeftCorner<3, 3>() = Quaterniond{x(3), x(0), x(1), x(2)}.toRotationMatrix();
  result.topRightCorner<3, 1>() = x.tail<3>();
  return result;
}

// The 4 by 4 matrix of the Lie algebra element of a pose difference (phi, rho).
Matrix4d hat(const Vector6d& d) {
  Matrix4d result = Matrix4d::Zero();
  result.topLeftCorner<3, 3>() << 0, -d(2), d(1), d(2), 0, -d(0), -d(1), d(0), 0;
  result.topRightCorner<3, 1>() = d.tail<3>();
  return result;
}

Vector7d somePose(double scale) {
  Vector7d x;
  x.head<4>() = UnitQuaternion::exponential(scale * Vector3d{0.4, -0.7, 0.9}).coeffs();
  x.tail<3>() = scale * Vector3d{1.5, -2.0, 0.5};
  return x;
}

void additive_groups_compose_by_addition() {
  const Vector3d a{1, 2, 3};
  const Vector3d b{0.5, -1, 4};
  VERIFY_IS_APPROX(Vector3d{compose<Position>(a, b)}, (a + b).eval());
  VERIFY_IS_APPROX(Vector3d{inverse<Position>(a)}, (-a).eval());
  VERIFY(Vector3d{identity<Position>()}.isZero());
  VERIFY_IS_APPROX(Vector3d{exp<Position>(a)}, a);
  VERIFY(Matrix3d{adjoint<Position>(a)}.isIdentity());
  VERIFY(checkGroup(Position{}, a, b, Vector3d{0.1, 0.2, -0.3}).worst() < tolerance);

  // A heading wraps: 3 rad and 1 rad compose to 4 - 2 pi.
  const Matrix<double, 1, 1> three{3.0};
  const Matrix<double, 1, 1> one{1.0};
  VERIFY(std::abs(compose(Heading{}, three, one)(0) - (4.0 - 2 * std::numbers::pi)) < tolerance);
  VERIFY(checkGroup(Heading{}, three, one, Matrix<double, 1, 1>{0.7}).worst() < tolerance);
}

void unit_quaternion_is_a_group() {
  const Vector4d a = UnitQuaternion::exponential(Vector3d{0.4, -0.7, 0.9}).coeffs();
  const Vector4d b = UnitQuaternion::exponential(Vector3d{-1.1, 0.3, 0.2}).coeffs();
  const Vector3d d{0.3, 0.5, -0.4};
  VERIFY(checkGroup(Attitude{}, a, b, d).worst() < tolerance);
  const Quaterniond product = Quaterniond{a} * Quaterniond{b};
  VERIFY_IS_APPROX(Vector4d{compose<Attitude>(a, b)}, product.coeffs());
  VERIFY_IS_APPROX(Matrix3d{adjoint<Attitude>(a)}, Quaterniond{a}.toRotationMatrix());
}

void composite_is_a_group_block_by_block() {
  using State = Composite<Attitude, Bias>;
  Vector7d a;
  a << UnitQuaternion::exponential(Vector3d{0.4, -0.7, 0.9}).coeffs(), 1, 2, 3;
  Vector7d b;
  b << UnitQuaternion::exponential(Vector3d{-1.1, 0.3, 0.2}).coeffs(), -1, 0.5, 2;
  Vector6d d;
  d << 0.3, 0.5, -0.4, 0.1, 0.2, 0.3;
  VERIFY(checkGroup(State{}, a, b, d).worst() < tolerance);
  const Matrix<double, 6, 6> ad = adjoint<State>(a);
  VERIFY(ad.bottomRightCorner(3, 3).isIdentity());
  VERIFY(ad.topRightCorner(3, 3).isZero());
}

void pose_matches_homogeneous_matrices() {
  const Vector7d a = somePose(1.0);
  const Vector7d b = somePose(-0.6);
  Vector6d d;
  d << 0.3, 0.5, -0.4, 1.0, -2.0, 0.7;

  VERIFY(homogeneous(compose<Pose>(a, b)).isApprox(homogeneous(a) * homogeneous(b), 1e-12));
  VERIFY(homogeneous(inverse<Pose>(a)).isApprox(homogeneous(a).inverse(), 1e-12));
  VERIFY(homogeneous(exp<Pose>(d)).isApprox(Matrix4d{hat(d).exp()}, 1e-12));
  VERIFY(homogeneous(plus<Pose>(a, d)).isApprox(homogeneous(a) * Matrix4d{hat(d).exp()}, 1e-12));
  // Ad_a d is the element whose matrix is a hat(d) a^-1.
  const Vector6d turned = adjoint<Pose>(a) * d;
  VERIFY(hat(turned).isApprox(homogeneous(a) * hat(d) * homogeneous(a).inverse(), 1e-12));
  VERIFY(checkGroup(Pose{}, a, b, d).worst() < 1e-11);

  // A position's rows are a step in the body frame: a step in them alone moves the position by R times it.
  Vector6d alone = Vector6d::Zero();
  alone.tail<3>() << 0.1, 0.2, 0.3;
  const Vector7d moved = plus<Pose>(a, alone);
  VERIFY_IS_APPROX(moved.head<4>().eval(), a.head<4>().eval());
  VERIFY_IS_APPROX((moved.tail<3>() - a.tail<3>()).eval(), (Quaterniond{a(3), a(0), a(1), a(2)} * alone.tail<3>()));
}

void reset_jacobian_is_the_right_jacobian() {
  const Vector7d x = somePose(1.0);
  Vector6d d;
  d << 0.3, 0.5, -0.4, 1.0, -2.0, 0.7;
  // By differences of plus() and minus(), as the reset is defined [covariance-reset].
  const Vector7d moved = plus<Pose>(x, d);
  Matrix<double, 6, 6> numeric;
  const double h = 1e-6;
  for (int k = 0; k < 6; ++k) {
    Vector6d up = d;
    Vector6d down = d;
    up(k) += h;
    down(k) -= h;
    const Vector7d upMoved = plus<Pose>(x, up);
    const Vector7d downMoved = plus<Pose>(x, down);
    numeric.col(k) = (Vector6d{minus<Pose>(upMoved, moved)} - Vector6d{minus<Pose>(downMoved, moved)}) / (2 * h);
  }
  const Matrix<double, 6, 6> reset = resetJacobian(Pose{}, x, d);
  VERIFY(reset.isApprox(numeric, 1e-8));
  VERIFY_IS_APPROX(reset, (Matrix<double, 6, 6>{rightJacobian<Pose>(d)}));
  VERIFY_IS_APPROX((Matrix<double, 6, 6>{leftJacobian<Pose>(d)}), (Matrix<double, 6, 6>{rightJacobian<Pose>(-d)}));
}

void run_time_sized_group() {
  const auto space = Body{Attitude{}, Velocity{}, Position{}, Feet{Foot{}, 2}};
  VERIFY_IS_EQUAL(valueDimension(space), 16);
  VERIFY_IS_EQUAL(differenceDimension(space), 15);
  VectorXd a(16);
  a << UnitQuaternion::exponential(Vector3d{0.4, -0.7, 0.9}).coeffs(), 1, 2, 3, -1, 0.5, 2, 0.3, 0.1, -0.2, 4, 5, 6;
  VectorXd b(16);
  b << UnitQuaternion::exponential(Vector3d{-1.1, 0.3, 0.2}).coeffs(), 0.2, -0.4, 1, 2, 2, -3, 1, 1, 1, -2, 0, 3;
  const VectorXd d = VectorXd::LinSpaced(15, -0.5, 0.6);
  VERIFY(checkGroup(space, a, b, d).worst() < 1e-11);

  // Each foot is turned as the position is: the pose of the attitude and one foot agrees with the whole.
  struct FootPose : ExtendedPose<Attitude, Foot> {};
  Vector7d footA;
  footA << a.head<4>(), a.tail<3>();
  Vector6d footD;
  footD << d.head<3>(), d.tail<3>();
  const VectorXd moved = plus(space, a, d);
  VERIFY_IS_APPROX(moved.tail<3>().eval(), (Vector7d{plus<FootPose>(footA, footD)}.tail<3>().eval()));
}

void planar_pose_under_a_heading_in_degrees() {
  const Vector3d a{30.0, 1.0, 2.0};
  const Vector3d b{100.0, -0.5, 0.3};
  const Vector3d d{20.0, 0.4, -0.2};
  VERIFY(checkGroup(PlanarPose{}, a, b, d).worst() < 1e-11);

  const double radians = 30.0 * std::numbers::pi / 180.0;
  const Rotation2Dd turn{radians};
  const Vector3d product = compose<PlanarPose>(a, b);
  VERIFY(std::abs(product(0) - 130.0) < tolerance);
  VERIFY_IS_APPROX(product.tail<2>().eval(), (a.tail<2>() + turn * b.tail<2>()).eval());
  // The heading's column of the adjoint is per degree: a degree of turn at a moves the world-frame step by
  // -[1]x p, per radian.
  const Matrix3d ad = adjoint<PlanarPose>(a);
  VERIFY(std::abs(ad(1, 0) - a(2) * std::numbers::pi / 180.0) < tolerance);
  VERIFY(std::abs(ad(2, 0) + a(1) * std::numbers::pi / 180.0) < tolerance);
}

}  // namespace

EIGEN_DECLARE_TEST(spaces_group) {
  CALL_SUBTEST_1(additive_groups_compose_by_addition());
  CALL_SUBTEST_1(unit_quaternion_is_a_group());
  CALL_SUBTEST_1(composite_is_a_group_block_by_block());
  CALL_SUBTEST_2(pose_matches_homogeneous_matrices());
  CALL_SUBTEST_2(reset_jacobian_is_the_right_jacobian());
  CALL_SUBTEST_2(run_time_sized_group());
  CALL_SUBTEST_2(planar_pose_under_a_heading_in_degrees());
}
