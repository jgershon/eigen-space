// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_EXTENDEDPOSE_SPACE_H
#define EIGEN_EXTENDEDPOSE_SPACE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Whether \a R has a rotation's members of its own: \c ActionDimension, the size of the vectors it turns,
 * and rotationMatrix(q, out), integratedRotation(d, out), and actionJacobian(v, out) (see Rotation).
 */
template <typename R>
concept HasRotationMembers = requires(const R& rotation, MatrixXd& m) {
  typename std::integral_constant<int, R::ActionDimension>;
  rotation.rotationMatrix(m.col(0), m.middleRows(0, 0));
  rotation.integratedRotation(m.col(0), m.middleRows(0, 0));
  rotation.actionJacobian(m.col(0), m.middleRows(0, 0));
};

/** \internal A Periodic space of one fixed coordinate, or a tag derived from one: a heading, which turns vectors of
 * the plane. Its period is one turn, whatever its unit.
 */
template <typename R>
concept IsPlanarHeading = IsPeriodic<R> && R::FixedValueDimension == 1 && R::ValueDimensionStep == 0;

/** \internal The size of the vectors that the rotation \a R turns. */
template <typename R>
struct ActionDimensionOf {
  static constexpr int value = 2;  ///< a heading turns vectors of the plane
};

template <HasRotationMembers R>
struct ActionDimensionOf<R> {
  static constexpr int value = R::ActionDimension;
};

/** \internal Radians per unit of the heading \a R: a period is one turn. */
template <typename R>
inline constexpr double headingRadiansPerUnit = 2 * EIGEN_PI / R::period();

/** \internal Writes the matrix by which the value \a q of the rotation \a rotation turns a vector through \a out. */
template <typename R, typename Q, typename Out>
void rotationMatrixInto(const R& rotation, const MatrixBase<Q>& q, const MatrixBase<Out>& out) {
  if constexpr (HasRotationMembers<R>) {
    rotation.rotationMatrix(q, out);
  } else {
    using T = typename Out::Scalar;
    const T angle = T(q(0)) * T(headingRadiansPerUnit<R>);
    const T c = numext::cos(angle);
    const T s = numext::sin(angle);
    writable(out) << c, -s, s, c;
  }
}

/** \internal Writes \f$ V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds \f$ of the rotation \a rotation through \a out: the
 * matrix that turns a vector's step into the vector's change over a step \a d of the rotation.
 */
template <typename R, typename D, typename Out>
void integratedRotationInto(const R& rotation, const MatrixBase<D>& d, const MatrixBase<Out>& out) {
  if constexpr (HasRotationMembers<R>) {
    rotation.integratedRotation(d, out);
  } else {
    using T = typename Out::Scalar;
    const T angle = T(d(0)) * T(headingRadiansPerUnit<R>);
    const T angle2 = angle * angle;
    T a = T(1) - angle2 / T(6);
    T b = angle / T(2) - angle * angle2 / T(24);
    if (angle2 >= T(1e-8)) {
      a = numext::sin(angle) / angle;
      b = (T(1) - numext::cos(angle)) / angle;
    }
    writable(out) << a, -b, b, a;
  }
}

/** \internal Writes through \a out the derivative of \f$ R(\mathrm{Exp}(\phi)) v \f$ in the rotation's step
 * \f$ \phi \f$ at zero: how a small step of the rotation \a rotation moves the vector \a v.
 */
template <typename R, typename V, typename Out>
void actionJacobianInto(const R& rotation, const MatrixBase<V>& v, const MatrixBase<Out>& out) {
  if constexpr (HasRotationMembers<R>) {
    rotation.actionJacobian(v, out);
  } else {
    using T = typename Out::Scalar;
    const T perUnit = T(headingRadiansPerUnit<R>);
    writable(out) << T(-v(1)) * perUnit, T(v(0)) * perUnit;
  }
}

/** \internal The nodes of the 12-point Gauss-Legendre rule on [0, 1], by which ExtendedPose integrates its right
 * Jacobian.
 */
inline constexpr double unitGaussLegendreNodes[12] = {0.0092196828766404,  0.04794137181476255, 0.11504866290284765,
                                                      0.20634102285669125, 0.3160842505009099,  0.43738329574426555,
                                                      0.56261670425573445, 0.6839157494990901,  0.79365897714330875,
                                                      0.88495133709715235, 0.95205862818523745, 0.9907803171233596};

/** \internal The weights of the nodes of unitGaussLegendreNodes. */
inline constexpr double unitGaussLegendreWeights[12] = {0.0235876681932559,  0.0534696629976592, 0.0800391642716731,
                                                        0.10158371336153295, 0.1167462682691774, 0.1245735229067014,
                                                        0.1245735229067014,  0.1167462682691774, 0.10158371336153295,
                                                        0.0800391642716731,  0.0534696629976592, 0.0235876681932559};

}  // namespace internal

/** \geometry_module \ingroup Geometry_Module
 * \brief A group that can head an ExtendedPose: a rotation, which turns vectors.
 *
 * QUnitQuaternion is one, turning vectors of three coordinates, and so is a Periodic space of one coordinate, a
 * heading in any unit, turning vectors of the plane. A rotation written from scratch is a fixed-size LieGroup with
 *  - <tt>static constexpr int ActionDimension</tt>, the size \f$ n \f$ of the vectors it turns;
 *  - <tt>rotationMatrix(q, out)</tt>, the \f$ n \f$ by \f$ n \f$ matrix \f$ R(q) \f$ by which the value \a q turns
 *    a vector, a homomorphism: \f$ R(a \cdot b) = R(a) R(b) \f$;
 *  - <tt>integratedRotation(d, out)</tt>, \f$ V(d) = \int_0^1 R(\mathrm{Exp}(s d)) \, ds \f$;
 *  - <tt>actionJacobian(v, out)</tt>, the \f$ n \f$ by \f$ k \f$ derivative of \f$ R(\mathrm{Exp}(\phi)) v \f$ in
 *    the rotation's step \f$ \phi \f$ of \f$ k \f$ coordinates, at zero,
 *
 * each const, in any scalar, writing through \a out as plus() does.
 */
template <typename R>
concept Rotation =
    LieGroup<R> && R::ValueDimensionStep == 0 && (internal::HasRotationMembers<R> || internal::IsPlanarHeading<R>);

/** \geometry_module \ingroup Geometry_Module
 * \brief A rotation followed by vectors that turn with it, as one Lie group: \f$ SE(3) \f$ with one vector,
 * \f$ SE_2(3) \f$ with two, and \f$ SE_k(3) \f$ with any number.
 *
 * The blocks are the user's tags, as in a Composite, and a Repeated may be among the vectors, so that the group's
 * size is known only at run time:
 * \code
 * struct Attitude : QUnitQuaternion<si::radian> {};
 * struct Velocity : QLinear<isq::velocity, 3> {};
 * struct Position : QLinear<isq::displacement, 3> {};
 * struct Body : ExtendedPose<Attitude, Velocity, Position> {};
 * using BodyInSI = In<Body, si::metre / si::second, si::metre>;
 * \endcode
 * It holds the values of <tt>Composite<Attitude, Velocity, Position></tt>, in that order, and names them by the same
 * paths. It differs in its product, a rigid motion after another,
 * \f[ (R_a, v_a) \cdot (R_b, v_b) = (R_a R_b, \; v_a + R_a v_b), \f]
 * applied to every vector alike, and so in its \f$ \boxplus \f$ and \f$ \boxminus \f$: with \f$ V \f$ the rotation's
 * integratedRotation(), which for a unit quaternion is the left Jacobian of SO(3),
 * \f[ (R, v_i) \boxplus (\phi, \rho_i) = (R \, \mathrm{Exp}(\phi), \; v_i + R \, V(\phi) \, \rho_i). \f]
 * A vector's difference rows are then a step of that vector in the body frame at the value: to first order
 * \f$ v_i + R \rho_i \f$. The extended Kalman filter on this space is the invariant extended Kalman filter of
 * \ref Barrau2017 "Barrau and Bonnabel (2017)", with the error they call left-invariant; \f$ SE_2(3) \f$ for an
 * inertial unit is that of Barrau and Bonnabel and of \ref Hartley2020 "Hartley et al. (2020)".
 *
 * The rotation \a R is a Rotation, and each of \a Vectors a linear block that holds whole vectors of the size the
 * rotation turns. Each vector keeps its own unit: the product turns it by a rotation matrix, which is a plain number,
 * so the blocks of the adjoint and the Jacobians are in the units of their rows over their columns, such as metres per
 * radian. The right Jacobian is integrated by a 12-point Gauss-Legendre rule over the adjoint. Ported from
 * EmrysFilter's groups-prototype.
 */
template <Rotation R, Space... Vectors>
class ExtendedPose : public Composite<R, Vectors...> {
  /** The Composite that holds the blocks. */
  using Base = Composite<R, Vectors...>;

  static constexpr int N = internal::ActionDimensionOf<R>::value;                       ///< the vectors' size
  static constexpr int Rv = R::FixedValueDimension;                                     ///< the rotation's values
  static constexpr int Rd = internal::DifferenceSpaceOf<R>::type::FixedValueDimension;  ///< and differences

  static_assert((isLinear<Vectors> && ...), "the blocks after the rotation are vectors: linear blocks");
  static_assert(((Vectors::FixedValueDimension % N == 0 && Vectors::ValueDimensionStep % N == 0) && ...),
                "each block after the rotation holds whole vectors of the size the rotation turns");

  /** A square matrix of the vectors' size, in \a T. */
  template <typename T>
  using Turn = Matrix<T, N, N>;

 public:
  using Base::Base;

  ExtendedPose() = default;

  static constexpr int VectorDimension = N;               ///< the size of each vector
  static constexpr int RotationValueDimension = Rv;       ///< the rotation's values, which come first
  static constexpr int RotationDifferenceDimension = Rd;  ///< the rotation's differences, which come first

  /** Writes through \a out, a square matrix of the vectors' size, the matrix by which the value \a x turns its
   * vectors' body-frame steps into the world frame: the rotation matrix of its rotation.
   */
  template <typename X, typename Out>
  void turn(const MatrixBase<X>& x, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    const Matrix<T, Rv, 1> q = x.template head<Rv>().template cast<T>();
    internal::rotationMatrixInto(rotation(), q, out);
  }

  /** \returns the rotation's block */
  const R& rotation() const { return this->template block<0>(); }

  /** \returns the number of vectors, the repetitions of a Repeated block counted each */
  Index vectorCount() const { return (this->valueDimension() - Rv) / N; }

  /** Writes \f$ x \boxplus d = x \cdot \mathrm{Exp}(d) \f$ through \a out, column by column. */
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Index m = vectorCount();
    for (Index j = 0; j < o.cols(); ++j) {
      const Matrix<T, Rv, 1> q = x.col(j).template head<Rv>().template cast<T>();
      const Matrix<T, Rd, 1> phi = d.col(j).template head<Rd>().template cast<T>();
      Matrix<T, Rv, 1> moved;
      internal::plusInto(rotation(), q, phi, moved);
      o.col(j).template head<Rv>() = moved;
      Turn<T> turned;
      Turn<T> integrated;
      internal::rotationMatrixInto(rotation(), q, turned);
      internal::integratedRotationInto(rotation(), phi, integrated);
      const Turn<T> step = turned * integrated;
      for (Index i = 0; i < m; ++i) {
        o.col(j).template segment<N>(Rv + i * N) = x.col(j).template segment<N>(Rv + i * N).template cast<T>() +
                                                   step * d.col(j).template segment<N>(Rd + i * N).template cast<T>();
      }
    }
  }

  /** Writes \f$ a \boxminus b = \mathrm{Log}(b^{-1} \cdot a) \f$ through \a out, column by column. */
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Index m = vectorCount();
    for (Index j = 0; j < o.cols(); ++j) {
      const Matrix<T, Rv, 1> qa = a.col(j).template head<Rv>().template cast<T>();
      const Matrix<T, Rv, 1> qb = b.col(j).template head<Rv>().template cast<T>();
      Matrix<T, Rd, 1> phi;
      internal::minusInto(rotation(), qa, qb, phi);
      o.col(j).template head<Rd>() = phi;
      Turn<T> turned;
      Turn<T> integrated;
      internal::rotationMatrixInto(rotation(), qb, turned);
      internal::integratedRotationInto(rotation(), phi, integrated);
      const Turn<T> back = integrated.inverse() * turned.transpose();
      for (Index i = 0; i < m; ++i) {
        o.col(j).template segment<N>(Rd + i * N) = back * (a.col(j).template segment<N>(Rv + i * N).template cast<T>() -
                                                           b.col(j).template segment<N>(Rv + i * N).template cast<T>());
      }
    }
  }

  /** Writes the product \f$ a \cdot b \f$ through \a out: the rotations' product, and each vector of \a b turned by
   * \a a's rotation and added to \a a's.
   */
  template <typename A, typename B, typename Out>
  void compose(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Matrix<T, Rv, 1> qa = a.template head<Rv>().template cast<T>();
    const Matrix<T, Rv, 1> qb = b.template head<Rv>().template cast<T>();
    Matrix<T, Rv, 1> q;
    internal::composeInto(rotation(), qa, qb, q);
    o.template head<Rv>() = q;
    Turn<T> turned;
    internal::rotationMatrixInto(rotation(), qa, turned);
    for (Index i = 0; i < vectorCount(); ++i) {
      o.template segment<N>(Rv + i * N) = a.template segment<N>(Rv + i * N).template cast<T>() +
                                          turned * b.template segment<N>(Rv + i * N).template cast<T>();
    }
  }

  /** Writes the inverse of \a a through \a out: the rotation's inverse, and each vector turned back and negated. */
  template <typename A, typename Out>
  void inverse(const MatrixBase<A>& a, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Matrix<T, Rv, 1> qa = a.template head<Rv>().template cast<T>();
    Matrix<T, Rv, 1> q;
    internal::inverseInto(rotation(), qa, q);
    o.template head<Rv>() = q;
    Turn<T> turned;
    internal::rotationMatrixInto(rotation(), qa, turned);
    for (Index i = 0; i < vectorCount(); ++i) {
      o.template segment<N>(Rv + i * N) = -(turned.transpose() * a.template segment<N>(Rv + i * N).template cast<T>());
    }
  }

  /** Writes the identity through \a out: the rotation's identity and zero vectors. */
  template <typename Out>
  void identity(const MatrixBase<Out>& out) const {
    auto& o = writable(out);
    o.setZero();
    internal::identityInto(rotation(), o.template head<Rv>());
  }

  /** Writes \f$ \mathrm{Exp}(d) \f$ through \a out: the rotation's exponential, and each vector's step turned by the
   * rotation's integratedRotation().
   */
  template <typename D, typename Out>
  void exp(const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Matrix<T, Rd, 1> phi = d.template head<Rd>().template cast<T>();
    Matrix<T, Rv, 1> q;
    internal::expInto(rotation(), phi, q);
    o.template head<Rv>() = q;
    Turn<T> integrated;
    internal::integratedRotationInto(rotation(), phi, integrated);
    for (Index i = 0; i < vectorCount(); ++i) {
      o.template segment<N>(Rv + i * N) = integrated * d.template segment<N>(Rd + i * N).template cast<T>();
    }
  }

  /** Writes \f$ \mathrm{Log}(a) \f$ through \a out, the inverse of exp(). */
  template <typename A, typename Out>
  void log(const MatrixBase<A>& a, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    const Matrix<T, Rv, 1> q = a.template head<Rv>().template cast<T>();
    Matrix<T, Rd, 1> phi;
    internal::logInto(rotation(), q, phi);
    o.template head<Rd>() = phi;
    Turn<T> integrated;
    internal::integratedRotationInto(rotation(), phi, integrated);
    const Turn<T> back = integrated.inverse();
    for (Index i = 0; i < vectorCount(); ++i) {
      o.template segment<N>(Rd + i * N) = back * a.template segment<N>(Rv + i * N).template cast<T>();
    }
  }

  /** Writes the adjoint \f$ \mathrm{Ad}_a \f$ through \a out, a square matrix over the differences: the rotation's
   * adjoint in its own rows; the rotation matrix in each vector's; and below the rotation's columns, for the vector
   * \f$ v_i \f$, \f$ -G(v_i) \, \mathrm{Ad}_R \f$ with \f$ G \f$ the rotation's actionJacobian(), which for a unit
   * quaternion is \f$ [v_i]_\times R \f$.
   */
  template <typename A, typename Out>
  void adjoint(const MatrixBase<A>& a, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    auto& o = writable(out);
    o.setZero();
    const Matrix<T, Rv, 1> q = a.template head<Rv>().template cast<T>();
    Matrix<T, Rd, Rd> own;
    internal::adjointInto(rotation(), q, own);
    o.template topLeftCorner<Rd, Rd>() = own;
    Turn<T> turned;
    internal::rotationMatrixInto(rotation(), q, turned);
    for (Index i = 0; i < vectorCount(); ++i) {
      const Matrix<T, N, 1> v = a.template segment<N>(Rv + i * N).template cast<T>();
      Matrix<T, N, Rd> action;
      internal::actionJacobianInto(rotation(), v, action);
      o.template block<N, Rd>(Rd + i * N, 0) = -(action * own);
      o.template block<N, N>(Rd + i * N, Rd + i * N) = turned;
    }
  }

  /** Writes the right Jacobian \f$ J_r(d) = \int_0^1 \mathrm{Ad}_{\mathrm{Exp}(-s d)} \, ds \f$ through \a out, by a
   * 12-point Gauss-Legendre rule, which is exact to rounding for steps of a turn or less.
   */
  template <typename D, typename Out>
  void rightJacobian(const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    using T = typename Out::Scalar;
    using Value = Matrix<T, valueDimensionAtCompileTime<Base>, 1, 0, maxValueDimensionAtCompileTime<Base>, 1>;
    using Difference = Matrix<T, valueDimensionAtCompileTime<DifferenceSpace<Base>>, 1, 0,
                              maxValueDimensionAtCompileTime<DifferenceSpace<Base>>, 1>;
    using Square = internal::MatrixOf<
        T, valueDimensionAtCompileTime<DifferenceSpace<Base>>, valueDimensionAtCompileTime<DifferenceSpace<Base>>,
        maxValueDimensionAtCompileTime<DifferenceSpace<Base>>, maxValueDimensionAtCompileTime<DifferenceSpace<Base>>>;
    auto& o = writable(out);
    const Index n = d.size();
    o.setZero();
    auto value = Value{};
    value.resize(this->valueDimension());
    auto scaled = Difference{};
    scaled.resize(n);
    auto term = Square{};
    term.resize(n, n);
    for (int k = 0; k < 12; ++k) {
      scaled = -T(internal::unitGaussLegendreNodes[k]) * d.template cast<T>();
      exp(scaled, value);
      adjoint(value, term);
      o += T(internal::unitGaussLegendreWeights[k]) * term;
    }
  }

  /** Writes the reset Jacobian at the step \a d through \a out: rightJacobian() of \a d, whatever the value
   * (design/decisions.md [covariance-reset]).
   */
  template <typename X, typename D, typename Out>
  void resetJacobian(const MatrixBase<X>&, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    rightJacobian(d, out);
  }

  /** Equal when every stored parameter and size is. */
  friend bool operator==(const ExtendedPose&, const ExtendedPose&) = default;
};

namespace internal {

template <typename R, typename... Vectors>
inline constexpr bool IsUntagged<ExtendedPose<R, Vectors...>>{true};

/** \internal Declared, never defined: matched by a pointer to an ExtendedPose or a tag derived from one. */
template <typename R, typename... Vectors>
std::tuple<R, Vectors...>* extendedPoseBlocks(const ExtendedPose<R, Vectors...>*);

/** \internal An ExtendedPose, or a tag derived from one. */
template <typename T>
concept IsExtendedPose = requires(const T* t) { extendedPoseBlocks(t); };

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_EXTENDEDPOSE_SPACE_H
