// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_TYPED_LLT_H
#define EIGEN_TYPED_LLT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \ingroup Cholesky_Module
 * \brief The Cholesky decomposition of a typed covariance, \f$ A = L L^T \f$, typed at its API.
 *
 * The factorization is LLT's own, on the matrix's coefficients (design/decisions.md [decomposition-boundaries]);
 * this class types what goes in and comes out. \a A must be a matrix over one space on both sides,
 * <tt>QMatrix<S, S></tt>, whose entry (i, j) has the unit \f$ u_i u_j \f$, as a covariance does; an operator
 * <tt>QMatrix<S, Dual<S>></tt> has no Cholesky factor of a meaning. Then
 *  - matrixL() is <tt>QMatrix<S, Linear<n>></tt>: \f$ L L^T \f$ is over \a S on both sides, and \f$ L^{-1} x \f$
 *    whitens a deviation \a x in \a S to plain numbers;
 *  - solve(b) for \a b with rows in \a S, \f$ A^{-1} b \f$, has rows in <tt>Dual<S></tt> and \a b's columns, so
 *    \f$ b^T A^{-1} b \f$ is dimensionless and \f$ P H^T S^{-1} \f$ is a gain. \a b is assigned into rows in \a S
 *    first, so its units are checked and converted.
 */
template <typename RowSpace, typename ColSpace, typename Scalar_, int Options_, int UpLo_>
  requires(internal::has_ruled_spaces<QMatrix<RowSpace, ColSpace, Scalar_, Options_>>)
class LLT<QMatrix<RowSpace, ColSpace, Scalar_, Options_>, UpLo_> {
 public:
  using MatrixType = QMatrix<RowSpace, ColSpace, Scalar_, Options_>;
  using Scalar = Scalar_;
  using RealScalar = typename NumTraits<Scalar>::Real;
  static constexpr int Size = MatrixType::RowsAtCompileTime;
  static constexpr int UpLo = UpLo_;
  /** The plain matrix LLT factors. */
  using PlainMatrix = Matrix<Scalar, Size, Size, 0, MatrixType::MaxRowsAtCompileTime, MatrixType::MaxColsAtCompileTime>;

  static_assert(internal::spaces_equal<RowSpace, ColSpace>::value,
                "a typed Cholesky decomposition takes a matrix over one space on both sides, such as a covariance "
                "QMatrix<S, S>, whose entries have the units u_i u_j");

  LLT() = default;

  /** The decomposition of \a a. */
  template <typename InputType>
  explicit LLT(const EigenBase<InputType>& a) {
    compute(a.derived());
  }

  /** Computes the decomposition of \a a, assigned into this matrix's spaces first. */
  template <typename InputType>
  LLT& compute(const EigenBase<InputType>& a) {
    const MatrixType typed = a.derived();
    m_llt.compute(Map<const PlainMatrix>(typed.data(), typed.rows(), typed.cols()));
    return *this;
  }

  /** \returns Success if the matrix was positive definite, otherwise NumericalIssue */
  ComputationInfo info() const { return m_llt.info(); }

  /** \returns an estimate of the reciprocal condition number, as LLT's */
  RealScalar rcond() const { return m_llt.rcond(); }

  /** \returns the lower triangular factor \f$ L \f$, typed <tt>QMatrix<S, Linear<n>></tt> */
  auto matrixL() const {
    QMatrix<RowSpace, Linear<Size>, Scalar> result;
    Map<PlainMatrix>(result.data(), result.rows(), result.cols()) = m_llt.matrixL();
    return result;
  }

  /** \returns \f$ A^{-1} b \f$, whose rows are in <tt>Dual<S></tt> and columns \a b's */
  template <typename Rhs>
  auto solve(const MatrixBase<Rhs>& b) const {
    using RhsCols = typename internal::spaces_of<Rhs>::ColSpace;
    using Given = QMatrix<RowSpace, RhsCols, Scalar>;
    using Result = QMatrix<Dual<ColSpace>, RhsCols, Scalar>;
    const Given typed = b;
    using PlainRhs =
        Matrix<Scalar, Given::RowsAtCompileTime, Given::ColsAtCompileTime, Given::IsRowMajor ? RowMajor : ColMajor,
               Given::MaxRowsAtCompileTime, Given::MaxColsAtCompileTime>;
    Result result(typed.rows(), typed.cols());
    Map<PlainRhs>(result.data(), result.rows(), result.cols()) =
        m_llt.solve(Map<const PlainRhs>(typed.data(), typed.rows(), typed.cols()));
    return result;
  }

  Index rows() const { return m_llt.rows(); }
  Index cols() const { return m_llt.cols(); }

 private:
  LLT<PlainMatrix, UpLo_> m_llt;
};

}  // namespace Eigen

#endif  // EIGEN_TYPED_LLT_H
