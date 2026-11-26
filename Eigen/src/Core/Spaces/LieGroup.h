// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_LIEGROUP_H
#define EIGEN_SPACES_LIEGROUP_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// The operations of a LieGroup (Spaces/Group.h) as free functions, in two forms. Given the space, as plus() and
// minus() are, they compute on plain coordinates. Given a typed operand, or the group as a template argument, they
// read and return typed matrices: a value of G is a QVector<G>, a difference a QVector<DifferenceSpace<G>>, and the
// adjoint and the Jacobians map differences to differences, QMatrix<D, Dual<D>>, so that entry (i, j) has the unit
// u_i / u_j [dual-spaces]. A typed operand is assigned into the space the operation expects, so the assignment rules
// check its units and convert them (design/decisions.md [heterogeneous-results]).

namespace Eigen {

namespace internal {

template <typename Xpr>
using RowSpaceOfXpr = typename spaces_of<std::remove_cvref_t<Xpr>>::RowSpace;

/** \internal A typed value of a fixed-size space: one column whose row space the space rules check, so that the
 * space can be read off the operand.
 */
template <typename A>
concept TypedPoint =
    MatrixExpression<A> && std::remove_cvref_t<A>::ColsAtCompileTime == 1 && has_ruled_spaces<std::remove_cvref_t<A>> &&
    FixedSizeSpace<RowSpaceOfXpr<A>> && Space<RowSpaceOfXpr<A>>;

/** \internal A typed value of a fixed-size Lie group. */
template <typename A>
concept TypedGroupValue = TypedPoint<A> && LieGroup<RowSpaceOfXpr<A>>;

/** \internal A value of the space \a G in \a Scalar. */
template <typename G, typename Scalar>
using PointOf = QVector<G, Scalar>;

/** \internal A difference of the space \a G in \a Scalar. */
template <typename G, typename Scalar>
using StepOf = QVector<DifferenceSpace<G>, Scalar>;

/** \internal A map from the differences of \a G to themselves, in \a Scalar: an adjoint or a Jacobian. */
template <typename G, typename Scalar>
using StepMapOf = QMatrix<DifferenceSpace<G>, Dual<DifferenceSpace<G>>, Scalar>;

/** \internal The coefficients of the plain object \a m as a plain matrix over the same memory, for the group
 * members, which compute on numbers.
 */
template <typename Derived>
auto bufferOf(PlainObjectBase<Derived>& m) {
  constexpr int rows = Derived::RowsAtCompileTime;
  constexpr int cols = Derived::ColsAtCompileTime;
  constexpr int order = cols == 1 ? ColMajor : rows == 1 ? RowMajor : (Derived::IsRowMajor ? RowMajor : ColMajor);
  return Map<Matrix<typename Derived::Scalar, rows, cols, order>>(m.data());
}

template <typename Derived>
auto bufferOf(const PlainObjectBase<Derived>& m) {
  constexpr int rows = Derived::RowsAtCompileTime;
  constexpr int cols = Derived::ColsAtCompileTime;
  constexpr int order = cols == 1 ? ColMajor : rows == 1 ? RowMajor : (Derived::IsRowMajor ? RowMajor : ColMajor);
  return Map<const Matrix<typename Derived::Scalar, rows, cols, order>>(m.data());
}

/** \internal A plain vector of \a n coordinates of \a S, of its values or (with \a Difference) its differences. */
template <bool Difference, typename S, typename Scalar>
auto plainVectorOf(const S& space) {
  using Rows = std::conditional_t<Difference, DifferenceSpace<S>, S>;
  using Result = MatrixOf<Scalar, valueDimensionAtCompileTime<Rows>, 1, maxValueDimensionAtCompileTime<Rows>, 1>;
  auto result = Result{};
  result.resize(Difference ? differenceDimension(space) : valueDimension(space));
  return result;
}

/** \internal A plain square matrix over the differences of \a S. */
template <typename S, typename Scalar>
auto plainStepMapOf(const S& space) {
  using Rows = DifferenceSpace<S>;
  constexpr int n = valueDimensionAtCompileTime<Rows>;
  constexpr int maxN = maxValueDimensionAtCompileTime<Rows>;
  auto result = MatrixOf<Scalar, n, n, maxN, maxN>{};
  result.resize(differenceDimension(space), differenceDimension(space));
  return result;
}

}  // namespace internal

/** \ingroup Spaces_Module
 * \returns the product \f$ a \cdot b \f$ of two values of the group \a space, in plain coordinates: \a b applied after
 * \a a, in \a a's frame. In Linear it is <tt>a + b</tt>; in UnitQuaternion the quaternion product; in an ExtendedPose
 * the composition of the two rigid motions.
 */
template <LieGroup S, typename A, typename B>
auto compose(const S& space, const MatrixBase<A>& a, const MatrixBase<B>& b) {
  eigen_assert(a.size() == valueDimension(space) && b.size() == valueDimension(space));
  auto result = internal::plainVectorOf<false, S, typename A::Scalar>(space);
  internal::composeInto(space, a, b, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the inverse \f$ a^{-1} \f$ of a value of the group \a space, in plain coordinates
 */
template <LieGroup S, typename A>
auto inverse(const S& space, const MatrixBase<A>& a) {
  eigen_assert(a.size() == valueDimension(space));
  auto result = internal::plainVectorOf<false, S, typename A::Scalar>(space);
  internal::inverseInto(space, a, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the identity of the group \a space in \a Scalar, in plain coordinates: zero in Linear, and (0, 0, 0, 1) in
 * UnitQuaternion
 */
template <typename Scalar = double, LieGroup S>
auto identity(const S& space) {
  auto result = internal::plainVectorOf<false, S, Scalar>(space);
  internal::identityInto(space, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ \mathrm{Exp}(d) \f$, the value of the group \a space that the difference \a d steps to from the
 * identity, in plain coordinates: <tt>plus(space, x, d)</tt> is <tt>compose(space, x, exp(space, d))</tt>
 */
template <LieGroup S, typename D>
auto exp(const S& space, const MatrixBase<D>& d) {
  eigen_assert(d.size() == differenceDimension(space));
  auto result = internal::plainVectorOf<false, S, typename D::Scalar>(space);
  internal::expInto(space, d, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ \mathrm{Log}(a) \f$, the difference that steps from the identity of the group \a space to \a a, in
 * plain coordinates: <tt>minus(space, a, b)</tt> is <tt>log(space, compose(space, inverse(space, b), a))</tt>
 */
template <LieGroup S, typename A>
auto log(const S& space, const MatrixBase<A>& a) {
  eigen_assert(a.size() == valueDimension(space));
  auto result = internal::plainVectorOf<true, S, typename A::Scalar>(space);
  internal::logInto(space, a, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the adjoint \f$ \mathrm{Ad}_a \f$ of the group \a space at the value \a a, in plain coordinates: the matrix
 * over its differences with \f$ a \cdot \mathrm{Exp}(d) = \mathrm{Exp}(\mathrm{Ad}_a d) \cdot a \f$, which takes a
 * step in the body frame at \a a to the same step in the world frame. The identity for a group under addition.
 */
template <LieGroup S, typename A>
auto adjoint(const S& space, const MatrixBase<A>& a) {
  eigen_assert(a.size() == valueDimension(space));
  auto result = internal::plainStepMapOf<S, typename A::Scalar>(space);
  internal::adjointInto(space, a, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the right Jacobian \f$ J_r(d) \f$ of the group \a space at the difference \a d, in plain coordinates:
 * \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(d) \cdot \mathrm{Exp}(J_r(d) \epsilon) \f$. The identity for a
 * group under addition.
 */
template <LieGroup S, typename D>
auto rightJacobian(const S& space, const MatrixBase<D>& d) {
  eigen_assert(d.size() == differenceDimension(space));
  auto result = internal::plainStepMapOf<S, typename D::Scalar>(space);
  internal::rightJacobianInto(space, d, result);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the left Jacobian \f$ J_l(d) = J_r(-d) \f$ of the group \a space at the difference \a d, in plain
 * coordinates: \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(J_l(d) \epsilon) \cdot \mathrm{Exp}(d) \f$
 */
template <LieGroup S, typename D>
auto leftJacobian(const S& space, const MatrixBase<D>& d) {
  return Eigen::rightJacobian(space, (-d).eval());
}

/** \ingroup Spaces_Module
 * \returns the product \f$ a \cdot b \f$ in the fixed-size group \a G, typed: <tt>compose<Pose>(a, b)</tt> is a
 * <tt>QVector<Pose></tt>. Each operand is assigned into a <tt>QVector<G></tt> first, so a plain one is accepted where
 * \a G has no units and a typed one converts into \a G's.
 */
template <FixedSizeSpace G, typename A, typename B>
  requires LieGroup<G>
auto compose(const MatrixBase<A>& a, const MatrixBase<B>& b) {
  using T = typename A::Scalar;
  const internal::PointOf<G, T> x = a;
  const internal::PointOf<G, T> y = b;
  internal::PointOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::composeInto(G{}, internal::bufferOf(x), internal::bufferOf(y), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the inverse in the fixed-size group \a G, typed: <tt>inverse<Pose>(a)</tt>.
 */
template <FixedSizeSpace G, typename A>
  requires LieGroup<G>
auto inverse(const MatrixBase<A>& a) {
  using T = typename A::Scalar;
  const internal::PointOf<G, T> x = a;
  internal::PointOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::inverseInto(G{}, internal::bufferOf(x), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the identity of the fixed-size group \a G in \a Scalar, typed: <tt>identity<Pose>()</tt>.
 */
template <FixedSizeSpace G, typename Scalar = double>
  requires LieGroup<G>
auto identity() {
  internal::PointOf<G, Scalar> result;
  auto out = internal::bufferOf(result);
  internal::identityInto(G{}, out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ \mathrm{Exp}(d) \f$ in the fixed-size group \a G, typed: <tt>exp<Pose>(d)</tt>, with \a d assigned
 * into <tt>QVector<DifferenceSpace<G>></tt> first.
 */
template <FixedSizeSpace G, typename D>
  requires LieGroup<G>
auto exp(const MatrixBase<D>& d) {
  using T = typename D::Scalar;
  const internal::StepOf<G, T> step = d;
  internal::PointOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::expInto(G{}, internal::bufferOf(step), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ \mathrm{Log}(a) \f$ in the fixed-size group \a G, typed: a <tt>QVector<DifferenceSpace<G>></tt>.
 */
template <FixedSizeSpace G, typename A>
  requires LieGroup<G>
auto log(const MatrixBase<A>& a) {
  using T = typename A::Scalar;
  const internal::PointOf<G, T> x = a;
  internal::StepOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::logInto(G{}, internal::bufferOf(x), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the adjoint of the fixed-size group \a G at \a a, typed: a map from the differences of \a G to themselves,
 * whose entry (i, j) has the unit of difference coordinate i over that of j. The block of a vector's rows under the
 * rotation's columns of an ExtendedPose is in, say, metres per radian.
 */
template <FixedSizeSpace G, typename A>
  requires LieGroup<G>
auto adjoint(const MatrixBase<A>& a) {
  using T = typename A::Scalar;
  const internal::PointOf<G, T> x = a;
  internal::StepMapOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::adjointInto(G{}, internal::bufferOf(x), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the right Jacobian of the fixed-size group \a G at \a d, typed as adjoint() is.
 */
template <FixedSizeSpace G, typename D>
  requires LieGroup<G>
auto rightJacobian(const MatrixBase<D>& d) {
  using T = typename D::Scalar;
  const internal::StepOf<G, T> step = d;
  internal::StepMapOf<G, T> result;
  auto out = internal::bufferOf(result);
  internal::rightJacobianInto(G{}, internal::bufferOf(step), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns the left Jacobian \f$ J_l(d) = J_r(-d) \f$ of the fixed-size group \a G at \a d, typed as adjoint() is.
 */
template <FixedSizeSpace G, typename D>
  requires LieGroup<G>
auto leftJacobian(const MatrixBase<D>& d) {
  using T = typename D::Scalar;
  const internal::StepOf<G, T> step = -d;
  return Eigen::rightJacobian<G>(step);
}

/** \ingroup Spaces_Module
 * \returns the product \f$ a \cdot b \f$ of two typed values of a fixed-size group, which is \a a's row space:
 * <tt>compose(x, y)</tt>.
 */
template <typename A, typename B>
  requires internal::TypedGroupValue<A>
auto compose(const A& a, const B& b) {
  return Eigen::compose<internal::RowSpaceOfXpr<A>>(a, b);
}

/** \ingroup Spaces_Module
 * \returns the inverse of a typed value of a fixed-size group, which is its row space.
 */
template <typename A>
  requires internal::TypedGroupValue<A>
auto inverse(const A& a) {
  return Eigen::inverse<internal::RowSpaceOfXpr<A>>(a);
}

/** \ingroup Spaces_Module
 * \returns \f$ \mathrm{Log}(a) \f$ of a typed value of a fixed-size group, which is its row space.
 */
template <typename A>
  requires internal::TypedGroupValue<A>
auto log(const A& a) {
  return Eigen::log<internal::RowSpaceOfXpr<A>>(a);
}

/** \ingroup Spaces_Module
 * \returns the adjoint at a typed value of a fixed-size group, which is its row space.
 */
template <typename A>
  requires internal::TypedGroupValue<A>
auto adjoint(const A& a) {
  return Eigen::adjoint<internal::RowSpaceOfXpr<A>>(a);
}

/** \ingroup Spaces_Module
 * \returns \f$ x \boxplus d \f$ for a typed value \a x of a fixed-size space, which is its row space, typed: \a d is
 * assigned into the space's differences first, so its units are checked and converted. For a group it is
 * <tt>compose(x, exp<S>(d))</tt>.
 */
template <typename X, typename D>
  requires internal::TypedPoint<X> && MatrixExpression<D>
auto plus(const X& x, const D& d) {
  using S = internal::RowSpaceOfXpr<X>;
  using T = typename X::Scalar;
  const internal::PointOf<S, T> value = x;
  const internal::StepOf<S, T> step = d;
  internal::PointOf<S, T> result;
  auto out = internal::bufferOf(result);
  internal::plusInto(S{}, internal::bufferOf(value), internal::bufferOf(step), out);
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ a \boxminus b \f$ for typed values of a fixed-size space, which is \a a's row space, typed: a
 * <tt>QVector<DifferenceSpace<S>></tt>.
 */
template <typename A, typename B>
  requires internal::TypedPoint<A> && MatrixExpression<B>
auto minus(const A& a, const B& b) {
  using S = internal::RowSpaceOfXpr<A>;
  using T = typename A::Scalar;
  const internal::PointOf<S, T> left = a;
  const internal::PointOf<S, T> right = b;
  internal::StepOf<S, T> result;
  auto out = internal::bufferOf(result);
  internal::minusInto(S{}, internal::bufferOf(left), internal::bufferOf(right), out);
  return result;
}

}  // namespace Eigen

#endif  // EIGEN_SPACES_LIEGROUP_H
