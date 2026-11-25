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
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

#include "main.h"

#include <Eigen/Units>

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

}  // namespace

EIGEN_DECLARE_TEST(units) {
  CALL_SUBTEST_1(typed_sizes_come_from_the_spaces());
  CALL_SUBTEST_1(typed_plain_objects_keep_their_spaces());
  CALL_SUBTEST_1(space_rules_compare_units_not_types());
  CALL_SUBTEST_1(space_rules_type_sums_products_and_transposes());
}
