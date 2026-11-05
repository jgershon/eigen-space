// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_UNITQUATERNION_SPACE_H
#define EIGEN_UNITQUATERNION_SPACE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \geometry_module \ingroup Geometry_Module
 * \brief An attitude as a space: a [unit quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
 * of four coordinates whose differences are three-coordinate rotation vectors in the body frame.
 *
 * Values are stored as (x, y, z, w), the order of <tt>Quaternion::coeffs()</tt>, so a
 * state's attitude block is exactly what a <tt>Quaternion</tt> stores and map() views it
 * as one; the product is Eigen's, Hamilton's. normalize() scales a value to unit norm with
 * \f$ w \ge 0 \f$, choosing one of the two quaternions of each rotation. Differences are rotation
 * vectors in Linear<3>, applied on the right, in the body frame:
 * \f[ q \boxplus \delta = q \otimes \mathrm{Exp}(\delta), \qquad
 *     q \boxminus p = \mathrm{Log}(p^{-1} \otimes q), \f]
 * the right \f$ \oplus \f$ and \f$ \ominus \f$ of \ref Sola2018 "Solà, Deray, and Atchuthan (2018)"
 * on SO(3), the group's \f$ \boxplus \f$ and \f$ \boxminus \f$ in the sense of
 * \ref Hertzberg2013 "Hertzberg, Wagner, Frese, and Schröder (2013)". The logarithm takes the short
 * way round, so a difference has norm at most \f$ \pi \f$. The reset Jacobian is the right Jacobian
 * of SO(3) in closed form (rightJacobian()), so a covariance is carried to a moved mean without
 * automatic differentiation (design/decisions.md [covariance-reset], [unit-quaternion]).
 *
 * Name the block with a tag, <tt>struct Attitude : UnitQuaternion {};</tt>, and view it in model
 * code with <tt>UnitQuaternion::map<Attitude>(space, x)</tt>. Use Repeated for several.
 */
struct UnitQuaternion {
  static constexpr int FixedValueDimension = 4;  ///< four coordinates, (x, y, z, w)
  static constexpr int ValueDimensionStep = 0;   ///< not resizable
  static constexpr int MaxValueDimension = 4;    ///< the value dimension, as for any fixed space

  /** \returns 4, the number of coordinates of a value */
  static constexpr Index valueDimension() { return 4; }

  /** \returns the difference space: rotation vectors, three plain coordinates */
  Linear<3> difference() const { return {}; }

  /** Scales each column of \a block, a value (x, y, z, w) per column, to unit norm with
   * \f$ w \ge 0 \f$, writing through the const reference as writable() does.
   */
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) const {
    using T = typename Derived::Scalar;
    auto& q = writable(block);
    for (Index j = 0; j < q.cols(); ++j) {
      const T sign = q(3, j) < T(0) ? T(-1) : T(1);
      q.col(j) *= sign / q.col(j).norm();
    }
  }

  /** Writes \f$ x \boxplus d = x \otimes \mathrm{Exp}(d) \f$ through \a out, column by column,
   * normalized.
   */
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    for (Index j = 0; j < o.cols(); ++j) {
      o.col(j) = (quaternionOf<T>(x.col(j)) * exponential(d.col(j).template cast<T>())).coeffs();
    }
    normalize(o);
  }

  /** Writes \f$ a \boxminus b = \mathrm{Log}(b^{-1} \otimes a) \f$ through \a out, column by
   * column.
   */
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    for (Index j = 0; j < o.cols(); ++j) {
      o.col(j) = logarithm(quaternionOf<T>(b.col(j)).conjugate() * quaternionOf<T>(a.col(j)));
    }
  }

  /** Writes the reset Jacobian at the step \a d through \a out: rightJacobian() of \a d, whatever
   * the value. See resetJacobian(space, x, d) for what it is and its sources.
   */
  template <typename X, typename D, typename Out>
  void resetJacobian(const MatrixBase<X>&, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    writable(out) = rightJacobian(d.template cast<T>());
  }

  /** \returns the unit quaternion of the rotation vector \a d, the exponential map of SO(3):
   * \f$ \mathrm{Exp}(\delta) = (\hat\delta \sin\tfrac\theta2, \cos\tfrac\theta2) \f$ with
   * \f$ \theta = |\delta| \f$, and its series below a small angle, where it stays
   * differentiable. A process model that integrates a body rate \f$ \omega \f$ over \f$ \Delta t \f$
   * writes <tt>map<Attitude>(space, y) = map<Attitude>(space, y) * exponential(omega * dt);</tt>.
   */
  template <typename Derived>
  static Quaternion<typename Derived::Scalar> exponential(const MatrixBase<Derived>& d) {
    using T = typename Derived::Scalar;
    const T angle2 = d.squaredNorm();
    if (angle2 < T(1e-12)) {
      const T scale = T(0.5) - angle2 / T(48);
      return Quaternion<T>{T(1) - angle2 / T(8), d(0) * scale, d(1) * scale, d(2) * scale};
    }
    const T angle = numext::sqrt(angle2);
    const T scale = T(numext::sin(T(angle / T(2))) / angle);
    return Quaternion<T>{numext::cos(T(angle / T(2))), d(0) * scale, d(1) * scale, d(2) * scale};
  }

  /** \returns the rotation vector of the unit quaternion \a q, the logarithm map of SO(3), the
   * short way round: \a q and \f$ -q \f$ give the same vector, of norm at most \f$ \pi \f$.
   */
  template <typename Derived>
  static Matrix<typename Derived::Scalar, 3, 1> logarithm(const QuaternionBase<Derived>& q) {
    using T = typename Derived::Scalar;
    const T sign = q.w() < T(0) ? T(-1) : T(1);
    const T w = sign * q.w();
    const Matrix<T, 3, 1> v = sign * q.vec();
    const T s2 = v.squaredNorm();
    if (s2 < T(1e-12)) {
      return v * T(T(2) / w);
    }
    const T s = numext::sqrt(s2);
    const T angle = T(2) * numext::atan2(s, w);
    return v * T(angle / s);
  }

  /** \returns the right Jacobian of SO(3) at the rotation vector \a d,
   * \f[ J_r(\delta) = I - \frac{1 - \cos\theta}{\theta^2} [\delta]_\times
   *     + \frac{\theta - \sin\theta}{\theta^3} [\delta]_\times^2, \qquad \theta = |\delta|, \f]
   * from \ref Sola2018 "Solà, Deray, and Atchuthan (2018)", eq. (143), with its series
   * \f$ I - \tfrac12 [\delta]_\times + \tfrac16 [\delta]_\times^2 \f$ below a small angle. It takes
   * a change of the step to the change of \f$ q \boxplus \delta \f$ seen from there, so it is
   * both this space's reset Jacobian and the Jacobian of plus() in its step.
   */
  template <typename Derived>
  static Matrix<typename Derived::Scalar, 3, 3> rightJacobian(const MatrixBase<Derived>& d) {
    using T = typename Derived::Scalar;
    auto skew = Matrix<T, 3, 3>{};
    skew << T(0), -d(2), d(1), d(2), T(0), -d(0), -d(1), d(0), T(0);
    const T angle2 = d.squaredNorm();
    if (angle2 < T(1e-8)) {
      return Matrix<T, 3, 3>::Identity() - T(0.5) * skew + (skew * skew) / T(6);
    }
    const T angle = numext::sqrt(angle2);
    return Matrix<T, 3, 3>::Identity() - T((T(1) - numext::cos(angle)) / angle2) * skew +
           T((angle - numext::sin(angle)) / (angle2 * angle)) * (skew * skew);
  }

  /** \returns the coordinates of the identity rotation, (0, 0, 0, 1) in this space's
   * (x, y, z, w) order, where <tt>mean << 1, 0, 0, 0</tt> would be a half turn about x.
   */
  template <typename Scalar = double>
  static Matrix<Scalar, 4, 1> identity() {
    return Matrix<Scalar, 4, 1>{Scalar(0), Scalar(0), Scalar(0), Scalar(1)};
  }

  /** \returns the coordinates of \a q in this space's order, (x, y, z, w): its
   * <tt>coeffs()</tt>, for writing a <tt>Quaternion</tt> into a mean without counting
   * where w goes.
   */
  template <typename Derived>
  static Matrix<typename Derived::Scalar, 4, 1> coefficientsOf(const QuaternionBase<Derived>& q) {
    return q.coeffs();
  }

  /** \returns \a block, four coordinates (x, y, z, w) in one column, as a
   * <tt>Map</tt> of a <tt>Quaternion</tt> over the same memory: writable where
   * \a block is, otherwise a map of a const quaternion. Nothing is copied, so the map must not
   * outlive the storage, and a non-unit quaternion written through it is canonicalized where the
   * library takes values from outside, by normalizeValues(). Copy it with
   * <tt>Quaterniond q = map(block);</tt>.
   *
   * \a block must have direct access with unit inner stride, such as a column, a segment, or a
   * whole vector; an unevaluated expression does not compile, and is to be evaluated first. A
   * plain matrix passed as a temporary does not compile either, since the map would dangle; a
   * temporary <tt>Block</tt>, <tt>Map</tt>, or <tt>Ref</tt> views storage that outlives it and is
   * accepted.
   */
  template <typename BlockType>
  static auto map(BlockType&& block) {
    using Derived = std::remove_cvref_t<BlockType>;
    static_assert(MatrixExpression<BlockType>, "map() takes an Eigen vector of four coordinates");
    static_assert(
        Derived::ColsAtCompileTime == 1 && (Derived::RowsAtCompileTime == 4 || Derived::RowsAtCompileTime == Dynamic),
        "map() takes one column of four coordinates (x, y, z, w)");
    static_assert((Derived::Flags & DirectAccessBit) && Derived::InnerStrideAtCompileTime == 1,
                  "map() needs coefficients in memory with unit stride; evaluate the expression first");
    static_assert(std::is_lvalue_reference_v<BlockType> || !std::derived_from<Derived, PlainObjectBase<Derived>>,
                  "map() of a temporary matrix would dangle; store it first");
    eigen_assert(block.size() == 4 && "a unit quaternion has four coordinates");
    using Scalar = typename Derived::Scalar;
    if constexpr (!std::is_const_v<std::remove_reference_t<BlockType>> && bool(Derived::Flags & LvalueBit)) {
      return Map<Quaternion<Scalar>>{block.derived().data()};
    } else {
      return Map<const Quaternion<Scalar>>{block.derived().data()};
    }
  }

  /** \returns map() of the block named by \a Path (a tag or a TagPath to a UnitQuaternion, or a
   * tag derived from one) in \a x, a whole state of \a space in one column:
   * <tt>UnitQuaternion::map<Attitude>(space, x)</tt>. Writable where \a x is, as for map(block).
   */
  template <typename Path, Space S, typename X>
  static auto map(const S& space, X&& x) {
    static_assert(std::derived_from<internal::PathTag<Path>, UnitQuaternion>,
                  "the path does not name a UnitQuaternion block");
    static_assert(!internal::RunsThroughRepeated<Path>,
                  "the path runs through a Repeated block; name one repetition with map<Path>(space, x, element)");
    if constexpr (internal::RunsThroughRepeated<Path>) {
      // Past the assertion: idx() would add its own message, and the start only gives the
      // call its usual type, so that the caller's line adds none either.
      return mapAt(space, std::forward<X>(x), Index{0});
    } else {
      return mapAt(space, std::forward<X>(x), idx<Path>(space).first());
    }
  }

  /** \returns map() of repetition \a element of the block named by \a Path in \a x, a whole state
   * of \a space: \a Path starts at a Repeated block, as for idx(space, element), so
   * <tt>UnitQuaternion::map<Fleet>(space, x, k)</tt> is the attitude of vehicle \a k in a
   * <tt>Repeated<UnitQuaternion></tt> named Fleet, and
   * <tt>UnitQuaternion::map<TagPath<Fleet, Attitude>>(space, x, k)</tt> its Attitude block.
   */
  template <typename Path, Space S, typename X>
  static auto map(const S& space, X&& x, Index element) {
    static_assert(namesRepetition<Path>(), "the path does not name a repeated UnitQuaternion block");
    return mapAt(space, std::forward<X>(x), idx<Path>(space, element).first());
  }

 private:
  /** \returns the quaternion of the coefficients (x, y, z, w) in \a coefficients, one column whose
   * size may be known only at run time, in the scalar type \a T. Eigen builds a quaternion only
   * from a vector whose size is four at compile time.
   */
  template <typename T, typename Derived>
  static Quaternion<T> quaternionOf(const MatrixBase<Derived>& coefficients) {
    const Matrix<T, 4, 1> fixed = coefficients.template cast<T>();
    return Quaternion<T>{fixed};
  }

  /** \returns whether one repetition of the path \a Path, which starts at a Repeated block, is a
   * UnitQuaternion: the repeated block itself for a bare tag, otherwise the path's last tag.
   */
  template <typename Path>
  static consteval bool namesRepetition() {
    using Tag = internal::PathTag<Path>;
    if constexpr (!std::is_same_v<internal::AsTagPath<Path>, TagPath<Tag>>) {
      return std::derived_from<Tag, UnitQuaternion>;
    } else if constexpr (internal::IsRepeated<Tag>) {
      return std::derived_from<internal::RepeatedBlock<Tag>, UnitQuaternion>;
    } else {
      return false;
    }
  }

  /** \returns map() of the four coordinates of \a x from \a start, after checking that \a x is
   * one whole state of \a space and not a temporary matrix.
   */
  template <Space S, typename X>
  static auto mapAt(const S& space, X&& x, Index start) {
    using Derived = std::remove_cvref_t<X>;
    static_assert(Derived::ColsAtCompileTime == 1, "map() takes one state in one column");
    static_assert(std::is_lvalue_reference_v<X> || !std::derived_from<Derived, PlainObjectBase<Derived>>,
                  "map() of a temporary matrix would dangle; store it first");
    eigen_assert(x.size() == Eigen::valueDimension(space) && "the state does not fit the space");
    return map(x.template segment<4>(start));
  }

 public:
  /** Spaces compare equal: a UnitQuaternion has no parameters. */
  friend bool operator==(const UnitQuaternion&, const UnitQuaternion&) = default;
};

}  // namespace Eigen

#endif  // EIGEN_UNITQUATERNION_SPACE_H
