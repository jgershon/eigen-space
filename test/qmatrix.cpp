// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// QMatrix and QArray as the primary plain objects, with Matrix and Array as aliases on dimensionless Linear spaces
// (design/decisions.md [spaces-are-shape], [q-names], [space-first]): the old spellings name the same types, the
// integer parameters still deduce through the alias, and a typed plain object derives its sizes from its spaces and
// keeps them through copies, eval(), and the plain-type helpers.

#include "main.h"

#include <type_traits>

namespace {

// A unit as integer exponents over length and time, standing in for a units library as in spaces_units.cpp.
struct TestUnit {
  int length = 0;
  int time = 0;

  friend constexpr TestUnit operator*(TestUnit a, TestUnit b) { return {a.length + b.length, a.time + b.time}; }
  friend constexpr TestUnit operator/(TestUnit a, TestUnit b) { return {a.length - b.length, a.time - b.time}; }
  friend constexpr bool operator==(TestUnit, TestUnit) = default;
};

constexpr TestUnit metre{1, 0};
constexpr TestUnit second{0, 1};
constexpr TestUnit metrePerSecond = metre / second;

struct Position : QLinear<metre, 3> {};
struct Velocity : QLinear<metrePerSecond, 3> {};
using State = Composite<Position, Velocity>;
struct Landmarks : QLinear<metre, Dynamic, 8> {};

// x' = F x: rows in State, columns in its dual, so that F(i, j) has the unit u_i / u_j [dual-spaces].
using Transition = QMatrix<State, Dual<State>>;
using StateVector = QVector<State>;

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
constexpr int deducedRows(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>&) {
  return Rows;
}

template <typename T>
struct IsIntegerShapedMatrix : std::false_type {};
template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct IsIntegerShapedMatrix<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : std::true_type {};

}  // namespace

void old_spellings_name_the_q_types() {
  static_assert(std::is_same_v<Matrix<float, 3, 1>, QMatrix<Linear<3>, Linear<1>, float>>);
  static_assert(std::is_same_v<Matrix3d, QMatrix<Linear<3>, Linear<3>>>);
  static_assert(std::is_same_v<MatrixXf, QMatrix<Linear<Dynamic>, Linear<Dynamic>, float>>);
  static_assert(std::is_same_v<Matrix<float, Dynamic, Dynamic, ColMajor, 4, 4>,
                               QMatrix<Linear<Dynamic, 4>, Linear<Dynamic, 4>, float, ColMajor>>);
  static_assert(std::is_same_v<ArrayXXi, QArray<Linear<Dynamic>, Linear<Dynamic>, int>>);
  static_assert(std::is_same_v<Array4f, QArray<Linear<4>, Linear<1>, float>>);
  // The default Options is computed from the spaces as it was from the sizes: a row vector is row-major.
  static_assert(std::is_same_v<QMatrix<Linear<1>, Linear<3>>, RowVector3d>);
  static_assert((int(RowVector3d::Options) & RowMajor) == RowMajor);
}

void integer_parameters_deduce_through_the_alias() {
  static_assert(deducedRows(Vector4d{}) == 4);
  VERIFY_IS_EQUAL(deducedRows(MatrixXd(2, 3)), int(Dynamic));
  static_assert(IsIntegerShapedMatrix<Matrix2f>::value && IsIntegerShapedMatrix<MatrixX3d>::value);
  // A specialization written against the integers sees only dimensionless Linear spaces.
  static_assert(!IsIntegerShapedMatrix<Transition>::value && !IsIntegerShapedMatrix<ArrayXXd>::value);
}

void typed_sizes_come_from_the_spaces() {
  static_assert(Transition::RowsAtCompileTime == 6 && Transition::ColsAtCompileTime == 6);
  static_assert(std::is_same_v<Transition::RowSpace, State> && std::is_same_v<Transition::ColSpace, Dual<State>>);
  static_assert(std::is_same_v<StateVector, QMatrix<State, Linear<1>>> && StateVector::IsVectorAtCompileTime);

  using LandmarkVector = QVector<Landmarks>;
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

template <typename Xpr>
using RowSpaceOf = typename internal::spaces_of<Xpr>::RowSpace;
template <typename Xpr>
using ColSpaceOf = typename internal::spaces_of<Xpr>::ColSpace;

// Two blocks in one unit, and a resizable block, against the leaves they flatten to.
struct Offset : QLinear<metre, 2> {};
using Positions = Composite<Position, Offset>;
using Covariance = QMatrix<State, State>;

void space_rules_compare_units_not_types() {
  // Tags, nesting, and the split into blocks don't change what the coordinates are measured in.
  static_assert(internal::spaces_equal<Position, QLinear<metre, 3>>::value);
  static_assert(internal::spaces_equal<Positions, QLinear<metre, 5>>::value);
  static_assert(internal::spaces_equal<Composite<Positions>, Composite<QLinear<metre, 1>, QLinear<metre, 4>>>::value);
  static_assert(internal::spaces_equal<State, Composite<QLinear<metre, 3>, QLinear<metrePerSecond, 3>>>::value);
  // A resizable run in a unit matches any number of coordinates in it, and no others.
  static_assert(internal::spaces_equal<Landmarks, Position>::value);
  static_assert(!internal::spaces_equal<Landmarks, Velocity>::value);
  static_assert(!internal::spaces_equal<Position, Velocity>::value);
  static_assert(!internal::spaces_equal<State, Composite<Velocity, Position>>::value);
  static_assert(!internal::spaces_equal<Position, Linear<3>>::value);
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

EIGEN_DECLARE_TEST(qmatrix) {
  CALL_SUBTEST_1(old_spellings_name_the_q_types());
  CALL_SUBTEST_1(integer_parameters_deduce_through_the_alias());
  CALL_SUBTEST_1(typed_sizes_come_from_the_spaces());
  CALL_SUBTEST_1(typed_plain_objects_keep_their_spaces());
  CALL_SUBTEST_1(space_rules_compare_units_not_types());
  CALL_SUBTEST_1(space_rules_type_sums_products_and_transposes());
}
