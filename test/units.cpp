// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Matrices over spaces with units, on mp-units (design/decisions.md [spaces-are-shape], [leaf-quantities],
// [dual-spaces], [heterogeneous-results]): a typed plain object derives its sizes from its spaces and keeps them
// through copies, eval(), and the plain-type helpers; sums, products, transposes, and assignments are checked by the
// units of their spaces; and a covariance prediction on typed matrices has the values of the plain one.

// Before main.h, whose forbidden-identifier macros (I, min, max) would break mp-units' headers.
#include <mp-units/systems/international.h>
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

#include "main.h"

#include <Eigen/Geometry>
#include <Eigen/Units>

#include <complex>
#include <type_traits>

namespace {

namespace si = mp_units::si;
namespace isq = mp_units::isq;

constexpr auto metrePerSecond = si::metre / si::second;

struct Position : QLinear<isq::length, 3> {};
struct Velocity : QLinear<isq::speed, 3> {};
struct Offset : QLinear<isq::length, 2> {};
struct Landmarks : QLinear<isq::length, Dynamic, 8> {};
using State = In<Composite<Position, Velocity>, si::metre, metrePerSecond>;

// x' = F x: rows in State, columns in its dual, so that F(i, j) has the unit u_i / u_j [dual-spaces].
using Transition = QMatrix<State, Dual<State>>;
using Covariance = QMatrix<State, State>;
using StateVector = QVector<State>;

template <typename Xpr>
using RowSpaceOf = typename internal::spaces_of<Xpr>::RowSpace;
template <typename Xpr>
using ColSpaceOf = typename internal::spaces_of<Xpr>::ColSpace;

void typed_sizes_come_from_the_spaces() {
  static_assert(Transition::RowsAtCompileTime == 6 && Transition::ColsAtCompileTime == 6);
  static_assert(std::is_same_v<Transition::RowSpace, State> && std::is_same_v<Transition::ColSpace, Dual<State>>);
  static_assert(std::is_same_v<StateVector, QMatrix<State, Linear<1>>> && StateVector::IsVectorAtCompileTime);

  using LandmarkVector = QVector<In<Landmarks, si::metre>>;
  static_assert(LandmarkVector::RowsAtCompileTime == Dynamic && LandmarkVector::MaxRowsAtCompileTime == 8);
  LandmarkVector landmarks(5);
  VERIFY_IS_EQUAL(landmarks.rows(), 5);
  VERIFY_IS_EQUAL(landmarks.cols(), 1);
}

void typed_plain_objects_keep_their_spaces() {
  static_assert(std::is_same_v<Transition::PlainObject, Transition>);
  static_assert(std::is_same_v<internal::plain_matrix_type<Transition>::type, Transition>);
  static_assert(std::is_same_v<internal::eval<Transition>::type, const Transition&>);
  static_assert(std::is_same_v<internal::spaces_of<const Transition>::RowSpace, State>);
  using TypedArray = QArray<State, Linear<1>>;
  static_assert(std::is_same_v<internal::plain_matrix_type<TypedArray>::type, TypedArray>);

  Transition f = Transition::Identity();
  f(0, 3) = 0.5;
  const Transition copy = f;
  static_assert(std::is_same_v<decltype(copy.eval()), const Transition&>);
  VERIFY_IS_EQUAL(copy(0, 3), 0.5);
  VERIFY_IS_EQUAL(copy(5, 5), 1.0);

  StateVector x = StateVector::Zero();
  x(3) = 2.0;
  const auto [px, py, pz, vx, vy, vz] = x;
  VERIFY_IS_EQUAL(vx, 2.0);
  VERIFY_IS_EQUAL(px + py + pz + vy + vz, 0.0);
}

void space_rules_compare_units_not_types() {
  using PositionM = In<Position, si::metre>;
  using VelocityM = In<Velocity, metrePerSecond>;
  // Tags, nesting, the split into blocks, and the spelling of a unit don't change what the coordinates measure.
  static_assert(internal::spaces_equal<PositionM, QLinear<si::metre, 3>>::value);
  static_assert(internal::spaces_equal<PositionM, QLinear<isq::length[si::metre], 3>>::value);
  static_assert(internal::spaces_equal<In<Composite<Position, Offset>, si::metre>, QLinear<si::metre, 5>>::value);
  static_assert(internal::spaces_equal<In<Composite<Composite<Position, Offset>>, si::metre>,
                                       Composite<QLinear<si::metre, 1>, QLinear<si::metre, 4>>>::value);
  static_assert(internal::spaces_equal<State, Composite<QLinear<si::metre, 3>, QLinear<metrePerSecond, 3>>>::value);
  static_assert(internal::spaces_equal<PositionM, QLinear<si::metre / si::second * si::second, 3>>::value);
  // A resizable run matches any number of coordinates of its quantity, and no others.
  static_assert(internal::spaces_equal<In<Landmarks, si::metre>, PositionM>::value);
  static_assert(!internal::spaces_equal<In<Landmarks, si::metre>, VelocityM>::value);
  static_assert(!internal::spaces_equal<PositionM, VelocityM>::value);
  static_assert(!internal::spaces_equal<State, Composite<VelocityM, PositionM>>::value);
  static_assert(!internal::spaces_equal<PositionM, Linear<3>>::value);
  // The dual inverts every unit, and a dimensionless space is its own dual.
  static_assert(internal::spaces_dual<Dual<State>, State>::value && internal::spaces_dual<State, Dual<State>>::value);
  static_assert(!internal::spaces_dual<State, State>::value);
  static_assert(internal::spaces_dual<Linear<3>, Linear<3>>::value);
}

void space_rules_type_sums_products_and_transposes() {
  Transition f = Transition::Identity();
  f.topRightCorner<3, 3>().diagonal().setConstant(0.1);  // position += dt * velocity, dt in seconds
  Covariance p = Covariance::Identity();
  StateVector x = StateVector::LinSpaced(6, 1.0, 6.0);

  // Transpose swaps; a product is over its operands' outer spaces, and a sum over its operands'.
  static_assert(std::is_same_v<RowSpaceOf<decltype(f.transpose())>, Dual<State>> &&
                std::is_same_v<ColSpaceOf<decltype(f.transpose())>, State>);
  static_assert(std::is_same_v<RowSpaceOf<decltype(f * x)>, State> &&
                std::is_same_v<ColSpaceOf<decltype(f * x)>, Linear<1>>);
  static_assert(std::is_same_v<RowSpaceOf<decltype(f * p)>, State> &&
                std::is_same_v<ColSpaceOf<decltype(f * p)>, State>);
  using Prediction = decltype(f * p * f.transpose());  // row-major by Eigen's product rule, so compare the spaces
  static_assert(std::is_same_v<RowSpaceOf<Prediction>, State> && std::is_same_v<ColSpaceOf<Prediction>, State>);
  static_assert(std::is_same_v<decltype(p + p)::PlainObject, Covariance>);
  // A scalar is dimensionless, on either side; Zero() and Constant() are in the spaces of the type they are called on.
  static_assert(std::is_same_v<decltype(2.0 * p)::PlainObject, Covariance>);
  static_assert(std::is_same_v<decltype(p / 2.0)::PlainObject, Covariance>);
  static_assert(std::is_same_v<decltype(p - Covariance::Zero())::PlainObject, Covariance>);
  // A plain operand is untyped, as before.
  static_assert(std::is_same_v<decltype(Matrix3d() * Matrix3d() + Matrix3d())::PlainObject, Matrix3d>);
  // A quadratic form: a covector in Dual<State> against a vector in State is dimensionless.
  using StateCovector = QVector<Dual<State>>;
  static_assert(std::is_same_v<decltype(StateCovector().transpose() * x)::PlainObject, Matrix<double, 1, 1>>);

  // The values are those of the same computation on plain matrices.
  const Matrix<double, 6, 6> fPlain = Map<const Matrix<double, 6, 6>>(f.data());
  const Matrix<double, 6, 1> xPlain = Map<const Matrix<double, 6, 1>>(x.data());
  StateVector predicted = f * x;
  predicted += 0.5 * x;
  predicted *= 2.0;
  VERIFY_IS_APPROX((Map<const Matrix<double, 6, 1>>(predicted.data())), (2.0 * (fPlain * xPlain + 0.5 * xPlain)));
  Covariance q = f * p * f.transpose() + p;
  VERIFY_IS_APPROX((Map<const Matrix<double, 6, 6>>(q.data())),
                   (fPlain * fPlain.transpose() + Matrix<double, 6, 6>::Identity()));
}

namespace ft = mp_units::international;

constexpr double footInMetres = 0.3048;

using PositionM = In<Position, si::metre>;
using PositionFt = In<Position, ft::foot>;
using StateFt = In<Composite<Position, Velocity>, ft::foot, ft::foot / si::second>;
using StateMixed = In<Composite<Position, Velocity>, ft::foot, metrePerSecond>;

// Assignment converts into the destination's units, as quantity<si::metre> q = 3 * ft does; .in<>() converts
// explicitly; a sum is in the common unit whichever operand comes first.
void conversion_follows_mp_units() {
  QVector<PositionFt> feet(1.0, 2.0, 3.0);
  const QVector<PositionM> metres = feet;
  VERIFY_IS_APPROX((Map<const Vector3d>(metres.data())), (footInMetres * Vector3d(1.0, 2.0, 3.0)));
  const QVector<PositionM> again = feet.in<PositionM>();
  VERIFY_IS_APPROX((Map<const Vector3d>(again.data())), (Map<const Vector3d>(metres.data())));
  QVector<PositionM> sum = metres;
  sum += feet;
  VERIFY_IS_APPROX((Map<const Vector3d>(sum.data())), (2.0 * Map<const Vector3d>(metres.data())));

  static_assert(std::is_same_v<decltype(metres + feet)::PlainObject, decltype(feet + metres)::PlainObject>);
  static_assert(internal::IsBoundSpace<RowSpaceOf<decltype(metres + feet)>>);  // the tag is shared, so it stays
  const QVector<PositionM> mixedSum = metres + feet;
  VERIFY_IS_APPROX((Map<const Vector3d>(mixedSum.data())), (2.0 * Map<const Vector3d>(metres.data())));
  // Where the operands' tags differ, the common space is untagged.
  using Untagged = QVector<QLinear<ft::foot, 3>>;
  static_assert(!internal::IsBoundSpace<RowSpaceOf<decltype(metres + Untagged())>>);
}

// A covariance converts entry by entry, (i, j) by the factors of row i and column j, and a prediction in feet has the
// values of the same prediction in metres, converted.
void covariance_prediction_in_feet_and_metres() {
  using TransitionFt = QMatrix<StateFt, Dual<StateFt>>;
  using CovarianceFt = QMatrix<StateFt, StateFt>;
  TransitionFt f = TransitionFt::Identity();
  f.topRightCorner<3, 3>().diagonal().setConstant(0.1);
  CovarianceFt p;
  for (Index i = 0; i < 6; ++i)
    for (Index j = 0; j < 6; ++j) p(i, j) = 1.0 / double(1 + i + j);

  const Covariance pMetres = p;
  VERIFY_IS_APPROX(pMetres(0, 1), footInMetres * footInMetres * p(0, 1));
  VERIFY_IS_APPROX(pMetres(4, 5), footInMetres * footInMetres * p(4, 5));
  const Transition fMetres = f;  // dimensionless position-position block, seconds in the velocity column
  VERIFY_IS_APPROX(fMetres(0, 3), f(0, 3));

  const CovarianceFt qFeet = f * p * f.transpose() + p;
  const Covariance qMetres = fMetres * pMetres * fMetres.transpose() + pMetres;
  const Covariance qConverted = qFeet;
  VERIFY_IS_APPROX((Map<const Matrix<double, 6, 6>>(qConverted.data())),
                   (Map<const Matrix<double, 6, 6>>(qMetres.data())));
}

// A product whose inner coordinates differ in unit converts the right operand's rows; a plain matrix times a
// position in metres is in metres, untagged, since the plain rows gain a dimension.
void products_convert_their_inner_coordinates() {
  Transition f = Transition::Identity();
  f.topRightCorner<3, 3>().diagonal().setConstant(0.1);
  QVector<StateMixed> x;
  x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  const QVector<State> xMetres = x;
  const QVector<State> direct = f * x;
  const QVector<State> converted = f * xMetres;
  VERIFY_IS_APPROX((Map<const Matrix<double, 6, 1>>(direct.data())),
                   (Map<const Matrix<double, 6, 1>>(converted.data())));

  const Matrix3d rotation = AngleAxisd(0.3, Vector3d::UnitZ()).toRotationMatrix();
  const QVector<PositionM> position(1.0, 2.0, 3.0);
  using Rotated = decltype(rotation * position);
  static_assert(!internal::IsBoundSpace<RowSpaceOf<Rotated>> && internal::isBoundSpace<RowSpaceOf<Rotated>>());
  const QVector<PositionM> rotated = rotation * position;
  VERIFY_IS_APPROX((Map<const Vector3d>(rotated.data())), (rotation * Vector3d(1.0, 2.0, 3.0)));
}

struct Heading : QRadiansPi<si::radian> {};
using Pose = In<Composite<Position, Heading>, si::metre>;

// Coordinate arithmetic is the space's own on a group under addition: a heading adds and subtracts modulo its period,
// and a covariance over the difference space of a pose with a heading types its prediction. A quaternion is not one
// (failtest/space_rule_quaternion_*.cpp).
void additive_groups_type_coordinate_arithmetic() {
  static_assert(internal::isAdditiveGroup<Pose>() && internal::isAdditiveGroup<DifferenceSpace<Pose>>());
  static_assert(internal::isAdditiveGroup<Dual<Pose>>() && !internal::isAdditiveGroup<UnitQuaternion>());
  static_assert(!internal::is_plain_space<UnitQuaternion> && internal::is_plain_space<Linear<4>>);

  using Differences = DifferenceSpace<Pose>;
  using PoseCovariance = QMatrix<Differences, Differences>;
  using PoseTransition = QMatrix<Differences, Dual<Differences>>;
  PoseTransition f = PoseTransition::Identity();
  f(0, 3) = 0.5;
  PoseCovariance p = PoseCovariance::Identity();
  const PoseCovariance q = f * p * f.transpose() + p;
  VERIFY_IS_APPROX(q(0, 0), 2.25);
  VERIFY_IS_APPROX(q(3, 3), 2.0);

  QVector<Pose> a(1.0, 2.0, 3.0, 3.0);
  QVector<Pose> b(0.0, 0.0, 0.0, -3.0);
  const QVector<Pose> difference = a - b;  // 6 radians: right modulo the period until normalized
  VERIFY_IS_APPROX(difference(3), 6.0);
}

// An mp-units quantity scales a matrix as a number does, and its unit multiplies the row space: a time step times a
// velocity is a displacement, in whatever unit of time.
void quantities_scale_matrices() {
  const QVector<PositionM> p(1.0, 2.0, 3.0);
  const QVector<In<Velocity, metrePerSecond>> v(10.0, 20.0, 30.0);
  const auto dt = 0.1 * si::second;
  static_assert(!internal::IsBoundSpace<RowSpaceOf<decltype(dt * v)>>);  // a speed times a time: the tag goes
  const QVector<PositionM> next = p + dt * v;
  VERIFY_IS_APPROX((Map<const Vector3d>(next.data())), Vector3d(2.0, 4.0, 6.0));
  const auto dtMilliseconds = 100.0 * si::milli<si::second>;
  const QVector<PositionM> again = p + v * dtMilliseconds;
  VERIFY_IS_APPROX((Map<const Vector3d>(again.data())), Vector3d(2.0, 4.0, 6.0));
  const QVector<In<Velocity, metrePerSecond>> speed = (next - p) / dt;
  VERIFY_IS_APPROX((Map<const Vector3d>(speed.data())), (Map<const Vector3d>(v.data())));
}

struct Phasors : QLinear<isq::voltage_phasor, 2> {};

// Units do not depend on the scalar being complex: a complex spec holds a complex scalar, adjoint() swaps the spaces,
// and the complex parts keep the units [complex-scalars].
void complex_scalars_keep_units() {
  using PhasorsV = In<Phasors, si::volt>;
  using Voltages = QVector<PhasorsV, std::complex<double>>;
  Voltages a;
  a << std::complex<double>(1.0, 2.0), std::complex<double>(3.0, -1.0);
  static_assert(std::is_same_v<RowSpaceOf<decltype(a.adjoint())>, Linear<1>> &&
                std::is_same_v<ColSpaceOf<decltype(a.adjoint())>, PhasorsV>);
  static_assert(std::is_same_v<RowSpaceOf<decltype(a.real())>, PhasorsV> &&
                std::is_same_v<RowSpaceOf<decltype(-a)>, PhasorsV>);
  const Voltages negated = -a.conjugate();
  VERIFY_IS_APPROX(negated(0), std::complex<double>(-1.0, 2.0));
  const auto power = (a.adjoint() * a).eval();  // volts squared, untagged
  VERIFY_IS_APPROX(power(0, 0).real(), 15.0);
  static_assert(internal::spaceHoldsScalar<PhasorsV, std::complex<double>>() &&
                !internal::spaceHoldsScalar<PhasorsV, double>());
  static_assert(!internal::spaceHoldsScalar<PositionM, std::complex<double>>() &&
                internal::spaceHoldsScalar<QLinear<si::volt, 2>, std::complex<double>>());
}

// SE_2(3) with units: the attitude's steps in radians, a velocity and a position that turn with it. In binds the
// vectors' units by dimension; the attitude's values stay plain numbers and its differences radians.
struct Attitude : QUnitQuaternion<si::radian> {};
struct BodyVelocity : QLinear<isq::velocity, 3> {};
struct BodyPosition : QLinear<isq::displacement, 3> {};
struct Body : ExtendedPose<Attitude, BodyVelocity, BodyPosition> {};
using BodySI = In<Body, metrePerSecond, si::metre>;
using BodyFt = In<Body, ft::foot / si::second, ft::foot>;
using BodyStep = DifferenceSpace<BodySI>;

// The group operations on typed vectors: exp() of a step in radians, metres per second, and metres is a value of the
// group, log() inverts it, and the adjoint maps steps to steps with its blocks in the units of their rows over their
// columns. A step in feet converts on the way in.
void group_operations_keep_units() {
  static_assert(LieGroup<BodySI> && !internal::isAdditiveGroup<BodySI>());
  static_assert(internal::spaces_equal<
                BodyStep, Composite<QLinear<si::radian, 3>, QLinear<metrePerSecond, 3>, QLinear<si::metre, 3>>>::value);
  static_assert(!internal::spaces_equal<BodyStep, Linear<9>>::value);
  using Value = QVector<BodySI>;
  using Step = QVector<BodyStep>;
  static_assert(std::is_same_v<decltype(exp<BodySI>(Step())), Value>);
  static_assert(std::is_same_v<decltype(log(Value())), Step>);
  static_assert(std::is_same_v<decltype(adjoint(Value())), QMatrix<BodyStep, Dual<BodyStep>>>);
  static_assert(std::is_same_v<decltype(rightJacobian<BodySI>(Step())), QMatrix<BodyStep, Dual<BodyStep>>>);

  Step xi;
  xi << 0.3, -0.2, 0.5, 1.0, 2.0, -0.5, 4.0, -3.0, 2.0;
  const Value x = exp<BodySI>(xi);
  const Step back = log(x);
  VERIFY_IS_APPROX((Map<const Matrix<double, 9, 1>>(back.data())), (Map<const Matrix<double, 9, 1>>(xi.data())));

  // The same step in feet and feet per second gives the same value once converted.
  const QVector<DifferenceSpace<BodyFt>> xiFt = xi;
  VERIFY_IS_APPROX(xiFt(3), xi(3) / footInMetres);
  VERIFY_IS_APPROX(xiFt(0), xi(0));
  const Value fromFeet = exp<BodySI>(xiFt);
  VERIFY_IS_APPROX((Map<const Matrix<double, 10, 1>>(fromFeet.data())), (Map<const Matrix<double, 10, 1>>(x.data())));

  // a Exp(d) = Exp(Ad_a d) a, typed: the product of the adjoint and a step is a step.
  const QMatrix<BodyStep, Dual<BodyStep>> ad = adjoint(x);
  const Step turned = ad * xi;
  const Value left = compose(x, exp<BodySI>(xi));
  const Value right = compose(exp<BodySI>(turned), x);
  const Step error = minus(left, right);
  VERIFY((Map<const Matrix<double, 9, 1>>(error.data())).norm() < 1e-12);
  const Value moved = plus(x, xi);
  VERIFY_IS_APPROX((Map<const Matrix<double, 10, 1>>(moved.data())), (Map<const Matrix<double, 10, 1>>(left.data())));
  const Value e = identity<BodySI>();
  const Value roundTrip = compose(x, inverse(x));
  VERIFY((Map<const Matrix<double, 9, 1>>(Step{minus(roundTrip, e)}.data())).norm() < 1e-12);
}

}  // namespace

EIGEN_DECLARE_TEST(units) {
  CALL_SUBTEST_1(typed_sizes_come_from_the_spaces());
  CALL_SUBTEST_1(typed_plain_objects_keep_their_spaces());
  CALL_SUBTEST_1(space_rules_compare_units_not_types());
  CALL_SUBTEST_1(space_rules_type_sums_products_and_transposes());
  CALL_SUBTEST_1(conversion_follows_mp_units());
  CALL_SUBTEST_1(covariance_prediction_in_feet_and_metres());
  CALL_SUBTEST_1(products_convert_their_inner_coordinates());
  CALL_SUBTEST_1(additive_groups_type_coordinate_arithmetic());
  CALL_SUBTEST_1(quantities_scale_matrices());
  CALL_SUBTEST_1(complex_scalars_keep_units());
  CALL_SUBTEST_1(group_operations_keep_units());
}
