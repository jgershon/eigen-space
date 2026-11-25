// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// QMatrix and QArray as the primary plain objects, with Matrix and Array as aliases on dimensionless Linear spaces
// (design/decisions.md [spaces-are-shape], [q-names], [space-first]): the old spellings name the same types, the
// integer parameters still deduce through the alias, and tags on dimensionless leaves shape a matrix without units,
// which needs nothing beyond Core. Matrices with units are tested in units.cpp.

#include "main.h"

#include <type_traits>

namespace {

// Tags on dimensionless leaves: a typed shape with no units, which Core alone supports.
struct Position : Linear<3> {};
struct Velocity : Linear<3> {};
using State = Composite<Position, Velocity>;
using Transition = QMatrix<State, Dual<State>>;

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

void dimensionless_tags_shape_a_matrix() {
  static_assert(std::is_same_v<Dual<State>, State>);
  static_assert(Transition::RowsAtCompileTime == 6 && std::is_same_v<Transition::RowSpace, State>);
  static_assert(std::is_same_v<Transition::PlainObject, Transition>);
  Transition f = Transition::Identity();
  const Transition g = f * f + f.transpose();
  VERIFY_IS_EQUAL(g(5, 5), 2.0);
}

EIGEN_DECLARE_TEST(qmatrix) {
  CALL_SUBTEST_1(old_spellings_name_the_q_types());
  CALL_SUBTEST_1(integer_parameters_deduce_through_the_alias());
  CALL_SUBTEST_1(dimensionless_tags_shape_a_matrix());
}
