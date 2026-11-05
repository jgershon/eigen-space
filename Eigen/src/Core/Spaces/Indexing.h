// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_INDEXING_H
#define EIGEN_SPACES_INDEXING_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// Resolving Eigen index sets against a space, the coordinates a path selects in values and in differences,
// inserting and erasing rows of vectors and symmetric matrices, and reading a symmetric matrix from its lower
// triangle. Everything here is internal.

namespace Eigen {

namespace internal {

/** \internal A column of resolved indices with compile-time size \a Size and bound \a MaxSize. */
template <int Size, int MaxSize>
using IndexVector = Matrix<Index, Size, 1, ColMajor, MaxSize, 1>;

/** \internal Compile-time size of the subset that \a Indices selects from a vector of size \a SourceSize. */
template <typename Indices, int SourceSize, int SourceMaxSize>
struct IndexedSizeOf {
 private:
  /// a vector of the source's size, to index with \a Indices
  using Probe = Matrix<int, SourceSize, 1, ColMajor, SourceMaxSize, 1>;

  /// what indexing it gives, whose compile-time rows are the subset's size
  using Expr = std::remove_cvref_t<decltype(std::declval<const Probe&>()(std::declval<const Indices&>()))>;

  static_assert(Expr::ColsAtCompileTime == 1, "the indices must select a set, not one index");

 public:
  static constexpr int size = Expr::RowsAtCompileTime;  ///< the subset's size, or Dynamic

  /// the subset's bound: its size if fixed, else Eigen's bound on the expression, else the source's
  static constexpr int maxSize = size != Dynamic                         ? size
                                 : Expr::MaxRowsAtCompileTime != Dynamic ? int(Expr::MaxRowsAtCompileTime)
                                                                         : SourceMaxSize;

  using type = IndexVector<size, maxSize>;  ///< the resolved indices' type
};

/** \internal The index vector \a Indices resolves to, sized by the subset it selects from a vector of size
 * \a SourceSize.
 */
template <typename Indices, int SourceSize, int SourceMaxSize>
using IndexedSize = typename IndexedSizeOf<Indices, SourceSize, SourceMaxSize>::type;

/** \internal \returns the concrete indices that \a indices selects from \c 0..n-1. Any index
 * type Eigen accepts works, including symbolic ones such as \c lastN(2). Eigen
 * asserts that every index is in range.
 */
template <int SourceSize, int SourceMaxSize, typename Indices>
IndexedSize<Indices, SourceSize, SourceMaxSize> resolveIndices(Index n, const Indices& indices) {
  using Identity = Matrix<Index, SourceSize, 1, ColMajor, SourceMaxSize, 1>;
  const Identity identity = Identity::LinSpaced(n, 0, n - 1);
  return identity(indices);
}

/** \internal \returns the \a count indices from \a start on, with compile-time size \a Size. */
template <int Size, int MaxSize>
IndexVector<Size, MaxSize> consecutiveIndices(Index start, Index count) {
  return IndexVector<Size, MaxSize>::LinSpaced(count, start, start + count - 1);
}

/** \internal \returns whether the concatenation of \a indices contains no repeats. Compares every pair
 * rather than sorting a copy, so it allocates nothing.
 */
template <typename... Vectors>
bool distinctIndices(const Vectors&... indices) {
  bool distinct = true;
  std::size_t position = 0;
  const auto seenBefore = [&](Index index, std::size_t before) {
    std::size_t k = 0;
    bool seen = false;
    const auto scan = [&](const auto& vector) {
      for (const Index other : vector) {
        if (k++ < before && other == index) {
          seen = true;
        }
      }
    };
    (scan(indices), ...);
    return seen;
  };
  const auto check = [&](const auto& vector) {
    for (const Index index : vector) {
      if (seenBefore(index, position)) {
        distinct = false;
      }
      ++position;
    }
  };
  (check(indices), ...);
  return distinct;
}

/** \internal \returns the indices in \c 0..n-1 not in \a given, ascending, in storage with the source's
 * bound, so a bounded source allocates nothing. Fixed-size when both \a SourceSize and the size
 * of \a given are known at compile time.
 */
template <int SourceSize, int SourceMaxSize, int GivenSize, int GivenMaxSize>
auto complementIndices(Index n, const IndexVector<GivenSize, GivenMaxSize>& given) {
  auto taken = Matrix<bool, SourceSize, 1, ColMajor, SourceMaxSize, 1>{};
  taken.setConstant(n, false);
  for (const Index i : given) {
    taken(i) = true;
  }
  constexpr int RestSize = SourceSize != Dynamic && GivenSize != Dynamic ? SourceSize - GivenSize : Dynamic;
  auto result = IndexVector<RestSize, RestSize == Dynamic ? SourceMaxSize : RestSize>{};
  result.resize(n - given.size());
  Index out = 0;
  for (Index i = 0; i < n; ++i) {
    if (!taken(i)) {
      result(out++) = i;
    }
  }
  return result;
}

/** \internal Coordinates selected from a space: their rows in an operand of values, such as a mean
 * or a sample, and in an operand of differences, such as a covariance. The two differ where a
 * block's differences have other sizes than its values.
 */
template <typename Values, typename Differences>
struct Coordinates {
  Values values;            ///< rows in an operand of values
  Differences differences;  ///< rows in an operand of differences
};

/** \internal \returns the coordinates of \a space selected by \a values and \a differences, index sets
 * Eigen accepts over its values and its differences, resolved
 */
template <typename S, typename ValueIndices, typename DifferenceIndices>
auto resolveCoordinates(const S& space, const ValueIndices& values, const DifferenceIndices& differences) {
  return Coordinates{resolveIndices<valueDimensionAtCompileTime<S>, maxValueDimensionAtCompileTime<S>>(
                         valueDimension(space), asIndexSet(values)),
                     resolveIndices<valueDimensionAtCompileTime<DifferenceSpace<S>>,
                                    maxValueDimensionAtCompileTime<DifferenceSpace<S>>>(differenceDimension(space),
                                                                                        asIndexSet(differences))};
}

/** \internal \returns the coordinates of the subspace named by \a Path within \a space, resolved */
template <typename Path, typename S>
auto pathCoordinates(const S& space) {
  return resolveCoordinates(space, idx<Path>(space), diffIdx<Path>(space));
}

/** \internal Inserts \a rows into the rows of \a matrix before row \a at. */
template <typename MatrixType, typename Derived>
void insertRows(MatrixType& matrix, Index at, const MatrixBase<Derived>& rows) {
  const Index n = matrix.rows();
  const Index k = rows.rows();
  eigen_assert(at >= 0 && at <= n && rows.cols() == matrix.cols());
  auto result = MatrixType{};
  result.resize(n + k, matrix.cols());
  result.topRows(at) = matrix.topRows(at);
  result.middleRows(at, k) = rows;
  result.bottomRows(n - at) = matrix.bottomRows(n - at);
  matrix = std::move(result);
}

/** \internal Removes \a count rows of \a matrix starting at row \a at. */
template <typename MatrixType>
void eraseRows(MatrixType& matrix, Index at, Index count) {
  const Index n = matrix.rows();
  eigen_assert(at >= 0 && count >= 0 && at + count <= n);
  auto result = MatrixType{};
  result.resize(n - count, matrix.cols());
  result.topRows(at) = matrix.topRows(at);
  result.bottomRows(n - at - count) = matrix.bottomRows(n - at - count);
  matrix = std::move(result);
}

/** \internal Inserts rows and columns into the symmetric \a matrix before index \a at: the block
 * \a block on the diagonal, and \a cross (new rows by existing columns) off it.
 */
template <typename MatrixType, typename BlockDerived, typename CrossDerived>
void insertSymmetric(MatrixType& matrix, Index at, const MatrixBase<BlockDerived>& block,
                     const MatrixBase<CrossDerived>& cross) {
  const Index n = matrix.rows();
  const Index k = block.rows();
  const Index after = n - at;
  eigen_assert(at >= 0 && at <= n && block.cols() == k && cross.rows() == k && cross.cols() == n);
  auto result = MatrixType{};
  result.resize(n + k, n + k);
  result.topLeftCorner(at, at) = matrix.topLeftCorner(at, at);
  result.topRightCorner(at, after) = matrix.topRightCorner(at, after);
  result.bottomLeftCorner(after, at) = matrix.bottomLeftCorner(after, at);
  result.bottomRightCorner(after, after) = matrix.bottomRightCorner(after, after);
  result.block(at, at, k, k) = block;
  result.block(at, 0, k, at) = cross.leftCols(at);
  result.block(at, at + k, k, after) = cross.rightCols(after);
  result.block(0, at, at, k) = cross.leftCols(at).transpose();
  result.block(at + k, at, after, k) = cross.rightCols(after).transpose();
  matrix = std::move(result);
}

/** \internal Removes \a count rows and columns of \a matrix starting at index \a at. */
template <typename MatrixType>
void eraseSymmetric(MatrixType& matrix, Index at, Index count) {
  const Index n = matrix.rows();
  const Index after = n - at - count;
  eigen_assert(at >= 0 && count >= 0 && after >= 0);
  auto result = MatrixType{};
  result.resize(n - count, n - count);
  result.topLeftCorner(at, at) = matrix.topLeftCorner(at, at);
  result.topRightCorner(at, after) = matrix.topRightCorner(at, after);
  result.bottomLeftCorner(after, at) = matrix.bottomLeftCorner(after, at);
  result.bottomRightCorner(after, after) = matrix.bottomRightCorner(after, after);
  matrix = std::move(result);
}

/** \internal \returns the symmetric matrix whose lower triangle is \a matrix, as a \a Result: copied in one
 * pass from a plain matrix, and evaluated first from any other expression.
 */
template <typename Result, typename Derived>
Result symmetricFromLower(const MatrixBase<Derived>& matrix) {
  if constexpr (std::is_base_of_v<PlainObjectBase<Derived>, Derived>) {
    Result result = matrix.template selfadjointView<Lower>();
    return result;
  } else {
    Result result = matrix;
    result.template triangularView<StrictlyUpper>() = result.transpose();
    return result;
  }
}

/** \internal \returns a lazy expression for the block of the symmetric matrix whose lower
 * triangle is \a matrix, at rows \a rows and columns \a cols. Only lower-triangle
 * entries of \a matrix are read, whatever order the indices are in. The expression
 * refers to its arguments and must not outlive them.
 */
template <int Rows, int MaxRows, int Cols, int MaxCols, typename Derived>
auto lowerSymmetricBlock(const MatrixBase<Derived>& matrix, const IndexVector<Rows, MaxRows>& rows,
                         const IndexVector<Cols, MaxCols>& cols) {
  using Scalar = typename Derived::Scalar;
  using Result = MatrixOf<Scalar, Rows, Cols, MaxRows, MaxCols>;
  return Result::NullaryExpr(rows.size(), cols.size(), [&matrix = matrix.derived(), &rows, &cols](Index i, Index j) {
    const Index r = rows(i);
    const Index c = cols(j);
    return r >= c ? matrix(r, c) : matrix(c, r);
  });
}

/** \internal \returns the block of the symmetric matrix whose lower triangle is \a matrix, at \a indices
 * on both axes, which may be any index set Eigen accepts: a coefficient for a single index,
 * otherwise a lazy expression that holds its resolved indices and refers to \a matrix, so it
 * must not outlive \a matrix. Only lower-triangle entries are read.
 */
template <int Size, int MaxSize, typename Derived, typename Indices>
auto lowerSymmetricSelection(const MatrixBase<Derived>& matrix, const Indices& indices) {
  if constexpr (std::is_integral_v<Indices>) {
    return matrix.derived().coeff(indices, indices);
  } else {
    using Scalar = typename Derived::Scalar;
    using Resolved = IndexedSize<Indices, Size, MaxSize>;
    using Result = MatrixOf<Scalar, Resolved::RowsAtCompileTime, Resolved::RowsAtCompileTime,
                            Resolved::MaxRowsAtCompileTime, Resolved::MaxRowsAtCompileTime>;
    Resolved resolved = resolveIndices<Size, MaxSize>(matrix.rows(), indices);
    const Index n = resolved.size();
    return Result::NullaryExpr(n, n, [&matrix = matrix.derived(), resolved = std::move(resolved)](Index i, Index j) {
      const Index r = resolved(i);
      const Index c = resolved(j);
      return r >= c ? matrix(r, c) : matrix(c, r);
    });
  }
}

/** \internal \returns the block of the symmetric matrix whose lower triangle is \a matrix, at rows \a rows
 * and columns \a cols, which may be any index sets Eigen accepts: a coefficient if both are
 * single indices, otherwise an evaluated matrix. Only lower-triangle entries are read.
 */
template <int Size, int MaxSize, typename Derived, typename Rows, typename Cols>
auto lowerSymmetricCross(const MatrixBase<Derived>& matrix, const Rows& rows, const Cols& cols) {
  if constexpr (std::is_integral_v<Rows> && std::is_integral_v<Cols>) {
    return rows >= cols ? matrix.derived().coeff(rows, cols) : matrix.derived().coeff(cols, rows);
  } else {
    const auto resolvedRows = resolveIndices<Size, MaxSize>(matrix.rows(), asIndexSet(rows));
    const auto resolvedCols = resolveIndices<Size, MaxSize>(matrix.rows(), asIndexSet(cols));
    using RowIndices = std::remove_cvref_t<decltype(resolvedRows)>;
    using ColIndices = std::remove_cvref_t<decltype(resolvedCols)>;
    using Result = MatrixOf<typename Derived::Scalar, RowIndices::RowsAtCompileTime, ColIndices::RowsAtCompileTime,
                            RowIndices::MaxRowsAtCompileTime, ColIndices::MaxRowsAtCompileTime>;
    return Result{lowerSymmetricBlock(matrix, resolvedRows, resolvedCols)};
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_INDEXING_H
