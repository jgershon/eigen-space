// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_SPACE_H
#define EIGEN_SPACES_SPACE_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

/** \defgroup Spaces_Module Spaces module
 * \ingroup Core_Module
 *
 * A space describes how the coordinates of a vector behave: whether they wrap or are plain
 * linear values, and how many there are (design/decisions.md [spaces-are-shape]). A fixed-size space is an
 * empty type and costs nothing. A plain vector of dimension \c N is <tt>Linear<N></tt>.
 *
 * Values and differences each live in a space. A space whose values need bringing into
 * canonical form has a normalize() member, applied to blocks whose rows are its coordinates
 * and whose columns are samples. Its differences live in its difference space, from
 * difference(), which is another space: an angle in <tt>[0, 360)</tt> has differences in
 * <tt>[-180, 180)</tt>, and a Linear coordinate's differences are plain vectors. Generic code works
 * with them through four free functions, which do nothing where there is nothing to do:
 *  - plus() applies an update, \f$ x \boxplus \delta \f$, by default
 *    <tt>normalizeValues(space, x + δ)</tt>;
 *  - minus() takes a residual, \f$ a \boxminus b \f$, by default
 *    <tt>normalizeDifferences(space, a - b)</tt>;
 *  - normalizeValues() and normalizeDifferences() bring a value or a difference that came
 *    from outside, such as a user's mean or a model's output, into canonical form.
 * A space whose updates and residuals aren't addition and subtraction, such as latitude and
 * longitude with differences in meters, defines its own plus() and minus() members. A space is
 * linear when none of this does anything (isLinear<>).
 *
 * The coordinate types are Linear and Periodic. Their parameters are template arguments, so a
 * fixed-size coordinate type is always an empty type, and the common periods are named:
 * RadiansPi, Radians2Pi, Degrees180, and Degrees360. There is no bounded coordinate type
 * [no-bounded-coordinates]: a quantity that can't be negative is estimated as the vector it is the length of, or
 * as a log, and a distribution that has a bounded support states it itself.
 *
 * Spaces compose: Composite concatenates them, and Repeated repeats a fixed-size block
 * a runtime number of times. Named subspaces are tag types derived from a space
 * (<tt>struct Heading : Degrees360<> {};</tt>).
 *
 * A space's valueDimension() is the number of coordinates of a value. Its differences may have
 * fewer, as a unit quaternion's four coordinates have three-coordinate rotation vectors as
 * differences, so a distribution over a space has two dimensions: valueDimension(), the length of a mean or a
 * sample, and differenceDimension(), the rows of a covariance and the degrees of freedom. Generic
 * code calls the free functions valueDimension(space) and differenceDimension(space); a space
 * implements only the member valueDimension(), and its difference dimension is its difference
 * space's [sizes-and-dimensions]. A fixed-size space is an empty type whose value dimension is a compile-time
 * constant. A resizable space (a Repeated block, a coordinate type with \c Dynamic
 * dimension, or a Composite with one) stores its value dimension. Any number of a Composite's
 * blocks may be resizable, at any depth of nested composites, such as a set of tracks and a set of
 * landmarks.
 *
 * A space with at most one resizable block can take its sizes from a length alone, so it has
 * resize(n). A space with several can't: the same length splits among the blocks in more than one way.
 *
 * idx<Path>(space) turns a path (a tag or a TagPath) into an Eigen index set from the
 * space's sizes. idx<S, Path>() gives the same index as a compile-time constant.
 *
 * Every space exposes \c FixedValueDimension (the coordinates of its fixed part),
 * \c ValueDimensionStep (nonzero if it is resizable: the coordinates of one element of its
 * resizable part), and \c MaxValueDimension (the largest valid value dimension, or
 * \c Dynamic if unbounded). Like Eigen's matrices, the coordinate types take a dimension
 * and a maximum dimension, and Repeated takes a maximum number of repetitions.
 *
 * The \f$ \boxplus \f$ and \f$ \boxminus \f$ operators, and the idea of writing an estimator
 * once over any state space that provides them, are those of
 * \ref Hertzberg2013 "Hertzberg, Wagner, Frese, and Schröder (2013)"; a Periodic coordinate is
 * the circle \f$ S^1 \f$ of their examples. See also
 * [directional statistics](https://en.wikipedia.org/wiki/Directional_statistics) for why a mean
 * of angles is not the mean of their values.
 *
 * \code
 * #include <Eigen/Core>
 * \endcode
 */

namespace internal {

/** \internal Whether \a S has a normalize() the library can call on an Eigen block. */
template <typename S>
concept HasNormalize = requires(const S& space, MatrixXd& block) { space.normalize(block.middleRows(0, 0)); };

/** \internal Whether \a S has its own \f$ x \boxplus \delta \f$: a plus(x, d, out) the library can call on
 * Eigen blocks.
 */
template <typename S>
concept HasPlus = requires(const S& space, MatrixXd& block) {
  space.plus(block.middleRows(0, 0), block.middleRows(0, 0), block.middleRows(0, 0));
};

/** \internal Whether \a S has its own \f$ a \boxminus b \f$: a minus(a, b, out) the library can call on
 * Eigen blocks.
 */
template <typename S>
concept HasMinus = requires(const S& space, MatrixXd& block) {
  space.minus(block.middleRows(0, 0), block.middleRows(0, 0), block.middleRows(0, 0));
};

/** \internal Whether \a S has a normalize() that HasNormalize rejects but that a near miss of its form would
 * call: a template that is not const, or that takes its block by non-const reference, so that a
 * temporary block does not bind.
 */
template <typename S>
concept NearMissNormalize = !HasNormalize<S> && requires(S& space, MatrixXd& block) { space.normalize(block); };

/** \internal Whether \a S has a plus() that HasPlus rejects but that a near miss of its form would call, as
 * for NearMissNormalize.
 */
template <typename S>
concept NearMissPlus = !HasPlus<S> && requires(S& space, MatrixXd& block) { space.plus(block, block, block); };

/** \internal Whether \a S has a minus() that HasMinus rejects but that a near miss of its form would call, as
 * for NearMissNormalize.
 */
template <typename S>
concept NearMissMinus = !HasMinus<S> && requires(S& space, MatrixXd& block) { space.minus(block, block, block); };

/** \internal Whether \a S's normalize(), if the library can see one, takes an Eigen block of any number of
 * columns: false for a normalize() that is not a template, so that its address can be taken, and
 * that HasNormalize rejects, such as one written for a single vector, and false for a template
 * that is a near miss of the form (NearMissNormalize). The library would otherwise take it as
 * missing and the values as plain. A member template that rejects every block, or a misspelled
 * name, can't be seen this way and still counts as missing.
 */
template <typename S>
concept NormalizeTakesBlocks = HasNormalize<S> || (!requires { &S::normalize; } && !NearMissNormalize<S>);

/** \internal Whether \a S's plus(), if the library can see one, takes Eigen blocks: as NormalizeTakesBlocks. */
template <typename S>
concept PlusTakesBlocks = HasPlus<S> || (!requires { &S::plus; } && !NearMissPlus<S>);

/** \internal Whether \a S's minus(), if the library can see one, takes Eigen blocks: as NormalizeTakesBlocks. */
template <typename S>
concept MinusTakesBlocks = HasMinus<S> || (!requires { &S::minus; } && !NearMissMinus<S>);

/** \internal Whether \a S names its difference space with a difference() member. */
template <typename S>
concept HasDifference = requires(const S& space) { space.difference(); };

/** \internal The difference space of \a S: \a S itself unless it has a difference() member. */
template <typename S>
struct DifferenceSpaceOf {
  using type = S;
};

template <HasDifference S>
struct DifferenceSpaceOf<S> {
  using type = std::remove_cvref_t<decltype(std::declval<const S&>().difference())>;
};

/** \internal Whether \a D can be the difference space of \a S: a regular type with a value dimension, fixed if \a S
 * is and bounded if \a S is. Its sizes may differ from \a S's. (Not the whole Space concept,
 * which would recurse through a space that is its own difference space.)
 */
template <typename D, typename S>
concept DifferenceSpaceFor = std::regular<D> && requires(const D& d) {
  { d.valueDimension() } -> std::convertible_to<Index>;
  typename std::integral_constant<int, D::FixedValueDimension>;
  typename std::integral_constant<int, D::ValueDimensionStep>;
  typename std::integral_constant<int, D::MaxValueDimension>;
  requires(D::ValueDimensionStep == 0) == (S::ValueDimensionStep == 0);
  requires(D::MaxValueDimension == Dynamic) == (S::MaxValueDimension == Dynamic);
};

/** \internal Whether the values and the differences of \a S have the same sizes. */
template <typename S>
inline constexpr bool hasEqualSizes = DifferenceSpaceOf<S>::type::FixedValueDimension == S::FixedValueDimension &&
                                      DifferenceSpaceOf<S>::type::ValueDimensionStep == S::ValueDimensionStep &&
                                      DifferenceSpaceOf<S>::type::MaxValueDimension == S::MaxValueDimension;

/** \internal The number of resizable blocks of \a S, at any depth of nested composites: none for a fixed
 * space and one for any other resizable space. A Composite counts its blocks' (the specialization
 * after IsComposite).
 */
template <typename S>
inline constexpr int resizableBlockCount = S::ValueDimensionStep == 0 ? 0 : 1;

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief A type usable as the space of a vector's coordinates.
 *
 * Linear, Periodic, Composite, Repeated, and tags derived from them satisfy it. A
 * space written from scratch provides:
 *  - \c FixedValueDimension, \c ValueDimensionStep, and \c MaxValueDimension (\c int) as
 *    constant expressions, consistent with each other: a fixed space (\c ValueDimensionStep
 *    zero) has \c MaxValueDimension equal to \c FixedValueDimension, and a resizable one has
 *    \c MaxValueDimension at least \c FixedValueDimension, or \c Dynamic;
 *  - valueDimension(), and resize(n) if it is resizable (a Composite with several resizable
 *    blocks has none);
 *  - if its values need normalizing, a const normalize() that takes any Eigen block by const
 *    reference and writes through it, rows being coordinates and columns samples;
 *  - if its differences are not plain values of the space itself, a const difference()
 *    returning its difference space: a regular type with \c FixedValueDimension,
 *    \c ValueDimensionStep, \c MaxValueDimension, and valueDimension(), fixed if the space is and
 *    bounded if the space is. A space with normalize()
 *    must have one, even if it returns Linear: whether differences wrap as values do is a
 *    choice to state. A space without one is its own difference space. Its sizes may differ
 *    from the space's, as a unit quaternion's differences are three-coordinate rotation
 *    vectors;
 *  - if \f$ x \boxplus \delta \f$ is not <tt>normalize(x + δ)</tt>, a const
 *    <tt>plus(x, d, out)</tt>, and if \f$ a \boxminus b \f$ is not <tt>a - b</tt> normalized in
 *    the difference space, a const <tt>minus(a, b, out)</tt>. Each takes Eigen blocks by const
 *    reference, columns paired, and writes the canonical result through \a out, which does not
 *    alias the inputs. A space has both or neither, since a residual taken by subtraction would
 *    not invert its own update, and with them a difference(). Customization is by these members
 *    only: the library calls Eigen::plus() and Eigen::minus() qualified, so a free plus() or
 *    minus() in the space's namespace is never found;
 *  - optionally, for a space with its own plus(), the reset Jacobian that carries a covariance to
 *    the tangent space at a moved mean (design/decisions.md [covariance-reset], Eigen/Spaces): a const
 *    <tt>resetJacobian(x, d, out)</tt> that writes it in closed form at one value column \a x and
 *    step \a d into the square \a out, or <tt>static constexpr bool identityReset{true};</tt> when
 *    it is the identity, as for a multiplicative \f$ x e^\delta \f$. Without either it is computed
 *    by automatic differentiation of plus() and minus(), which must then take
 *    \c AutoDiffScalar coefficients;
 *  - value semantics: default construction, copying, and equality (\c std::regular).
 *
 * A normalize(), plus(), or minus() that is not a template and can't be called on Eigen blocks,
 * such as one written for a single vector, keeps the type from being a Space, whether it is on
 * the space or on its difference space: the library calls each on blocks of many columns, such
 * as every particle's state at once. So does a template that is a near miss of the form, one that
 * is not const or that takes a block by non-const reference, which the library would otherwise
 * take as missing and the space as a plain vector. One the library can't see, such as a
 * misspelled name or a member template that rejects every block, counts as missing.
 *
 * A space written from scratch is a single block: a tag path can name it but not reach
 * inside it.
 */
template <typename S>
concept Space = requires(const S& space) {
  typename std::integral_constant<int, S::FixedValueDimension>;
  typename std::integral_constant<int, S::ValueDimensionStep>;
  typename std::integral_constant<int, S::MaxValueDimension>;
  requires S::FixedValueDimension >= 0 && S::ValueDimensionStep >= 0;
  requires(S::ValueDimensionStep == 0
               ? S::MaxValueDimension == S::FixedValueDimension
               : (S::MaxValueDimension == Dynamic || S::MaxValueDimension >= S::FixedValueDimension));
  requires std::regular<S>;
  { space.valueDimension() } -> std::convertible_to<Index>;
  requires internal::NormalizeTakesBlocks<S> && internal::PlusTakesBlocks<S> && internal::MinusTakesBlocks<S>;
  requires internal::NormalizeTakesBlocks<typename internal::DifferenceSpaceOf<S>::type>;
  requires internal::HasDifference<S> ||
               (!internal::HasNormalize<S> && !internal::HasPlus<S> && !internal::HasMinus<S>);
  requires internal::DifferenceSpaceFor<typename internal::DifferenceSpaceOf<S>::type, S>;
  requires internal::HasPlus<S> == internal::HasMinus<S>;
  requires internal::hasEqualSizes<S> || internal::HasPlus<S>;
  requires S::ValueDimensionStep == 0 || internal::resizableBlockCount<S> > 1 ||
                   requires(S& resizable, Index n) { resizable.resize(n); };
};

/** \ingroup Spaces_Module
 * A space whose instance holds nothing, so that <tt>S{}</tt> is the only one there is: a space
 * of fixed size, which is an empty type (design/decisions.md [space-storage]). The free functions that take a space,
 * such as plus() and minus(), take one of these as a template argument instead: <tt>minus<State>(a, b)</tt>.
 * A fixed-size space that stores state is not one, and is passed as an instance.
 */
template <typename S>
concept FixedSizeSpace = std::is_empty_v<S> && std::default_initializable<S>;

/** \ingroup Spaces_Module
 * The space that differences of values in \a S live in: the type of its difference(), or \a S
 * itself if it has none.
 */
template <Space S>
using DifferenceSpace = typename internal::DifferenceSpaceOf<S>::type;

/** \ingroup Spaces_Module
 * \returns the difference space of \a space, with its sizes: its difference(), or a copy of
 * \a space if it is its own difference space. A fixed-size one is an empty type.
 */
template <Space S>
DifferenceSpace<S> differenceSpace(const S& space) {
  if constexpr (internal::HasDifference<S>) {
    return space.difference();
  } else {
    return space;
  }
}

/** \ingroup Spaces_Module
 * \returns the number of coordinates of a value in \a space, the length of a mean or a sample:
 * the space's own valueDimension().
 *
 * It is for run time and is not \c constexpr, even for a fixed space, as Eigen has \c rows() beside
 * \c RowsAtCompileTime: a resizable space's dimension is never a constant, and a user's space would
 * need every member \c constexpr. At compile time use valueDimensionAtCompileTime, which works for
 * every space, and <tt>valueDimensionAtCompileTime<DifferenceSpace<S>></tt> for the differences.
 */
template <Space S>
Index valueDimension(const S& space) {
  return space.valueDimension();
}

/** \ingroup Spaces_Module
 * \returns the number of coordinates of a difference in \a space, the rows of a covariance, the
 * \f$ d \f$ in a density's normalizer, and the degrees of freedom of a residual: the value dimension
 * of its difference space.
 */
template <Space S>
Index differenceDimension(const S& space) {
  if constexpr (internal::HasDifference<S>) {
    return space.difference().valueDimension();
  } else {
    return space.valueDimension();
  }
}

/** \ingroup Spaces_Module
 * Whether the values of \a S are plain: it has neither normalize() nor plus(), so
 * \f$ x \boxplus \delta \f$ is <tt>x + δ</tt>. Periodic's are not; a space that wraps only its
 * differences has plain values.
 */
template <Space S>
inline constexpr bool hasPlainValues = !internal::HasNormalize<S> && !internal::HasPlus<S>;

/** \ingroup Spaces_Module
 * Whether the differences of \a S are plain: its difference space has no normalize(), and it
 * has neither plus() nor minus(), so \f$ a \boxminus b \f$ is <tt>a - b</tt> and
 * \f$ x \boxplus \delta \f$ is <tt>x + δ</tt> up to normalize(). Linear has plain differences,
 * so a weighted mean in it is the plain one; Periodic does not.
 */
template <Space S>
inline constexpr bool hasPlainDifferences =
    !internal::HasNormalize<DifferenceSpace<S>> && !internal::HasPlus<S> && !internal::HasMinus<S>;

/** \ingroup Spaces_Module
 * Whether \a S is linear: its values and its differences are both plain vectors. Linear, and a
 * Composite or Repeated of linear blocks, are linear; Periodic is not. Like hasPlainValues and
 * hasPlainDifferences, it takes only a Space, so a type that is not one does not compile rather
 * than counting as linear.
 */
template <Space S>
inline constexpr bool isLinear = hasPlainValues<S> && hasPlainDifferences<S>;

/** \ingroup Spaces_Module
 * \brief \a block, writable: Eigen's idiom for a function that writes through an expression
 * passed as <tt>const&</tt>.
 *
 * A space's normalize(), plus(), and minus() take their output as a
 * <tt>const MatrixBase<Derived>&</tt>, so that a temporary block such as
 * <tt>x.segment(i, n)</tt> can be passed, and write through it with this:
 * <tt>writable(block) = block.unaryExpr(f)</tt>.
 */
template <typename Derived>
MatrixBase<Derived>& writable(const MatrixBase<Derived>& block) {
  return const_cast<MatrixBase<Derived>&>(block);
}

/** \ingroup Spaces_Module
 * Brings the values in \a block (rows are \a space's coordinates, columns samples) into
 * canonical form, writing through the const reference as the space's normalize() does: for
 * values that come from outside, such as a user's mean or a model's output. Does nothing for a
 * space without normalize(). An update goes through plus() instead.
 */
template <Space S, typename Derived>
void normalizeValues(const S& space, const MatrixBase<Derived>& block) {
  if constexpr (internal::HasNormalize<S>) {
    space.normalize(block);
  }
}

/** \ingroup Spaces_Module
 * Brings the differences in \a block into canonical form: normalizeValues() in the difference
 * space of \a space, for differences that come from outside. Does nothing where that space has
 * no normalize(). A residual goes through minus() instead.
 */
template <Space S, typename Derived>
void normalizeDifferences(const S& space, const MatrixBase<Derived>& block) {
  if constexpr (internal::HasNormalize<DifferenceSpace<S>>) {
    differenceSpace(space).normalize(block);
  }
}

/** \ingroup Spaces_Module
 * normalizeValues() in the fixed-size space \a S, given as a template argument:
 * <tt>normalizeValues<State>(block)</tt>.
 */
template <FixedSizeSpace S, typename Derived>
  requires Space<S>
void normalizeValues(const MatrixBase<Derived>& block) {
  normalizeValues(S{}, block);
}

/** \ingroup Spaces_Module
 * normalizeDifferences() in the fixed-size space \a S, given as a template argument:
 * <tt>normalizeDifferences<State>(block)</tt>.
 */
template <FixedSizeSpace S, typename Derived>
  requires Space<S>
void normalizeDifferences(const MatrixBase<Derived>& block) {
  normalizeDifferences(S{}, block);
}

namespace internal {

/** \internal \returns \a x wrapped into <tt>[Lower, Lower + Period)</tt>.
 *
 * Values almost always arrive in range or within one period of it (a canonical value plus a small
 * update), so those cases take a comparison or one addition; only values further out pay for the
 * division and floor.
 */
template <double Lower, double Period, typename Scalar>
constexpr Scalar wrapValue(Scalar x) {
  constexpr double upper = Lower + Period;
  if (x >= Scalar(Lower) && x < Scalar(upper)) [[likely]] {
    return x;
  }
  auto y = Scalar{};
  if (x >= Scalar(Lower - Period) && x < Scalar(upper + Period)) [[likely]] {
    // Not ?:, whose arms are distinct expression types for a scalar such as AutoDiffScalar.
    if (x < Scalar(Lower)) {
      y = x + Scalar(Period);
    } else {
      y = x - Scalar(Period);
    }
  } else if constexpr (requires {
                         x.value();
                         x.derivatives();
                       }) {
    // An AutoDiffScalar, which has no floor(): the count of periods is constant where it is differentiable.
    using Value = std::remove_cvref_t<decltype(x.value())>;
    y = x - Value(Period) * numext::floor((x.value() - Value(Lower)) / Value(Period));
  } else {
    y = x - Scalar(Period) * numext::floor((x - Scalar(Lower)) / Scalar(Period));
  }
  // Rounding can leave y an ulp or so below the lower bound: the quotient can round up to the next
  // integer, and Lower - Period is itself rounded. One more period brings it back. Just below the
  // lower bound, that sum, or the one above, can round up to the excluded upper bound. Written so
  // that NaN passes through.
  if (y < Scalar(Lower)) {
    y += Scalar(Period);
  }
  return y >= Scalar(upper) ? Scalar(Lower) : y;
}

/** \internal A plain matrix with the storage order Eigen requires for single-row matrices. */
template <typename Scalar, int Rows, int Cols, int MaxRows, int MaxCols>
using MatrixOf = Matrix<Scalar, Rows, Cols, (MaxRows == 1 && MaxCols != 1 ? RowMajor : ColMajor), MaxRows, MaxCols>;

/** \internal How the columns of \a L and \a R pair up in plus() and minus(): an operand with one column at
 * compile time is used for every column of the other.
 */
template <typename L, typename R>
struct PairedColumns {
  /// \a L has one column at compile time and \a R more: use it for every column of \a R
  static constexpr bool broadcastLeft = L::ColsAtCompileTime == 1 && R::ColsAtCompileTime != 1;

  /// \a R has one column at compile time and \a L more: use it for every column of \a L
  static constexpr bool broadcastRight = R::ColsAtCompileTime == 1 && L::ColsAtCompileTime != 1;

  /// the result's columns at compile time
  static constexpr int cols = broadcastLeft ? int(R::ColsAtCompileTime) : int(L::ColsAtCompileTime);

  /// the result's bound on its columns
  static constexpr int maxCols = broadcastLeft ? int(R::MaxColsAtCompileTime) : int(L::MaxColsAtCompileTime);

  /** \returns the number of paired columns of \a left and \a right */
  template <typename Left, typename Right>
  static Index count(const Left& left, const Right& right) {
    eigen_assert((broadcastLeft || broadcastRight || left.cols() == right.cols()) &&
                 "the operands have different numbers of columns");
    return broadcastLeft ? right.cols() : left.cols();
  }
};

/** \internal \returns \a m with \a cols columns: repeated if \a Broadcast, otherwise \a m itself. */
template <bool Broadcast, typename Derived>
decltype(auto) withColumns(const MatrixBase<Derived>& m, Index cols) {
  if constexpr (Broadcast) {
    return m.derived().replicate(1, cols);
  } else {
    return m.derived();
  }
}

/** \internal \returns \a m itself where its coefficients are in memory (a plain matrix, a map, or a block of
 * one), and otherwise \a m evaluated into a plain matrix: what a space's plus() or minus() reads
 * by block or by column without recomputing an expression such as a product.
 */
template <typename Derived>
decltype(auto) inMemory(const MatrixBase<Derived>& m) {
  if constexpr ((traits<Derived>::Flags & DirectAccessBit) != 0) {
    return m.derived();
  } else {
    return m.eval();
  }
}

/** \internal Writes \f$ x \boxplus \delta \f$ through \a out: \a space's plus(), or <tt>x + d</tt>
 * normalized.
 */
template <typename S, typename X, typename D, typename Out>
void plusInto(const S& space, const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) {
  if constexpr (HasPlus<S>) {
    space.plus(x, d, out);
  } else {
    writable(out) = x + d;
    normalizeValues(space, out);
  }
}

/** \internal Writes \f$ a \boxminus b \f$ through \a out: \a space's minus(), or <tt>a - b</tt> normalized
 * in the difference space.
 */
template <typename S, typename A, typename B, typename Out>
void minusInto(const S& space, const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) {
  if constexpr (HasMinus<S>) {
    space.minus(a, b, out);
  } else {
    writable(out) = a - b;
    normalizeDifferences(space, out);
  }
}

/** \internal The number of rows \a space takes in an operand of differences if \a Difference, otherwise of
 * values.
 */
template <bool Difference, typename S>
Index rowsOf(const S& space) {
  if constexpr (Difference) {
    return differenceSpace(space).valueDimension();
  } else {
    return space.valueDimension();
  }
}

/** \internal The number of rows rowsOf() gives for a space of type \a S at compile time, or \c Dynamic. */
template <bool Difference, typename S>
inline constexpr int rowsAtCompileTime =
    Difference ? valueDimensionAtCompileTime<DifferenceSpace<S>> : valueDimensionAtCompileTime<S>;

/** \internal \returns the \a rows rows of \a m from \a offset, as a block of \a Rows rows where \a Rows is
 * fixed, so that Eigen unrolls what a block of a fixed-size space does with them, and of
 * \a rows rows otherwise.
 */
template <int Rows, typename M>
auto middleRowsOf(M& m, Index offset, Index rows) {
  if constexpr (Rows == Dynamic) {
    return m.middleRows(offset, rows);
  } else {
    eigen_assert(rows == Rows && "the block's rows must be its compile-time size");
    return m.template middleRows<Rows>(offset);
  }
}

/** \internal \returns the \a cols columns of \a m from \a offset, as a block of \a Cols columns where
 * \a Cols is fixed, as middleRowsOf() gives rows.
 */
template <int Cols, typename M>
auto middleColsOf(M& m, Index offset, Index cols) {
  if constexpr (Cols == Dynamic) {
    return m.middleCols(offset, cols);
  } else {
    eigen_assert(cols == Cols && "the block's columns must be its compile-time size");
    return m.template middleCols<Cols>(offset);
  }
}

/** \internal \returns the \a rows by \a cols block of \a m from row \a row and column \a col, with \a Rows
 * rows and \a Cols columns at compile time where they are fixed, as middleRowsOf() gives rows.
 */
template <int Rows, int Cols, typename M>
auto blockOf(M& m, Index row, Index col, Index rows, Index cols) {
  eigen_assert((Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols) &&
               "the block's sizes must be its compile-time sizes");
  return m.template block<Rows, Cols>(row, col, rows, cols);
}

}  // namespace internal

/** \ingroup Spaces_Module
 * \returns \f$ x \boxplus \delta \f$: the values \a x moved by the differences \a d, in canonical
 * form. Rows are coordinates and columns samples; an operand with one column at compile time is
 * used for every column of the other, so <tt>plus(space, mean, L * z)</tt> draws samples. For
 * most spaces this is <tt>normalizeValues(space, x + d)</tt>, and for a linear one just
 * <tt>x + d</tt>.
 */
template <Space S, typename XDerived, typename DDerived>
auto plus(const S& space, const MatrixBase<XDerived>& x, const MatrixBase<DDerived>& d) {
  using Columns = internal::PairedColumns<XDerived, DDerived>;
  // The result's rows are the space's where it knows them, so that an unbounded operand into a
  // bounded space gives a bounded result.
  constexpr int rows =
      valueDimensionAtCompileTime<S> != Dynamic ? valueDimensionAtCompileTime<S> : int(XDerived::RowsAtCompileTime);
  constexpr int maxRows = maxValueDimensionAtCompileTime<S> != Dynamic ? maxValueDimensionAtCompileTime<S>
                                                                       : int(XDerived::MaxRowsAtCompileTime);
  using Result = internal::MatrixOf<typename XDerived::Scalar, rows, Columns::cols, maxRows, Columns::maxCols>;
  const Index cols = Columns::count(x, d);
  auto result = Result{};
  result.resize(x.rows(), cols);
  // Evaluated once where they are expressions, so a space's plus() can read them by block or by
  // column without recomputing a product; a broadcast stays a lazy replicate of the evaluated operand.
  const auto& xInMemory = internal::inMemory(x);
  const auto& dInMemory = internal::inMemory(d);
  const auto& paired = internal::withColumns<Columns::broadcastLeft>(xInMemory, cols);
  const auto& offsets = internal::withColumns<Columns::broadcastRight>(dInMemory, cols);
  if constexpr (internal::HasPlus<S>) {
    space.plus(paired, offsets, result);
  } else {
    result = paired + offsets;
    normalizeValues(space, result);
  }
  return result;
}

/** \ingroup Spaces_Module
 * \returns \f$ a \boxminus b \f$: the differences from \a b to \a a, in canonical form. Rows are
 * coordinates and columns samples; an operand with one column at compile time is used for every
 * column of the other, so <tt>minus(space, samples, mean)</tt> gives each sample's deviation.
 * For most spaces this is <tt>normalizeDifferences(space, a - b)</tt>, and for a linear one just
 * <tt>a - b</tt>.
 */
template <Space S, typename ADerived, typename BDerived>
auto minus(const S& space, const MatrixBase<ADerived>& a, const MatrixBase<BDerived>& b) {
  using Columns = internal::PairedColumns<ADerived, BDerived>;
  // The result's rows are the difference space's where it knows them, so that an unbounded
  // operand into a bounded space gives a bounded result, and the operands' rows otherwise, which
  // is right only where the space's differences have the sizes of its values.
  constexpr bool sameRows = internal::hasEqualSizes<S>;
  constexpr int spaceRows = valueDimensionAtCompileTime<DifferenceSpace<S>>;
  constexpr int spaceMaxRows = maxValueDimensionAtCompileTime<DifferenceSpace<S>>;
  constexpr int rows = spaceRows != Dynamic ? spaceRows : sameRows ? int(ADerived::RowsAtCompileTime) : Dynamic;
  constexpr int maxRows = spaceMaxRows != Dynamic ? spaceMaxRows
                          : sameRows              ? int(ADerived::MaxRowsAtCompileTime)
                                                  : Dynamic;
  using Result = internal::MatrixOf<typename ADerived::Scalar, rows, Columns::cols, maxRows, Columns::maxCols>;
  const Index cols = Columns::count(a, b);
  auto result = Result{};
  result.resize(sameRows ? a.rows() : differenceDimension(space), cols);
  const auto& aInMemory = internal::inMemory(a);
  const auto& bInMemory = internal::inMemory(b);
  const auto& left = internal::withColumns<Columns::broadcastLeft>(aInMemory, cols);
  const auto& right = internal::withColumns<Columns::broadcastRight>(bInMemory, cols);
  if constexpr (internal::HasMinus<S>) {
    space.minus(left, right, result);
  } else {
    result = left - right;
    normalizeDifferences(space, result);
  }
  return result;
}

/** \ingroup Spaces_Module
 * \returns plus() in the fixed-size space \a S, given as a template argument:
 * <tt>plus<State>(x, d)</tt>.
 */
template <FixedSizeSpace S, typename XDerived, typename DDerived>
  requires Space<S>
auto plus(const MatrixBase<XDerived>& x, const MatrixBase<DDerived>& d) {
  return Eigen::plus(S{}, x, d);
}

/** \ingroup Spaces_Module
 * \returns minus() in the fixed-size space \a S, given as a template argument:
 * <tt>minus<State>(a, b)</tt>.
 */
template <FixedSizeSpace S, typename ADerived, typename BDerived>
  requires Space<S>
auto minus(const MatrixBase<ADerived>& a, const MatrixBase<BDerived>& b) {
  return Eigen::minus(S{}, a, b);
}

/** \ingroup Spaces_Module
 * \returns whether some instance of \a S has value dimension \a n: at least its
 * \c FixedValueDimension, a whole number of \c ValueDimensionStep beyond it, and at most its
 * \c MaxValueDimension. An instance's own is its valueDimension().
 *
 * The test is exact for a space with at most one resizable block. For one with several it is
 * necessary but not sufficient: a Composite of blocks with strides 2 and 3 has stride 1, and no
 * instance has 1 coordinate.
 */
template <typename S>
constexpr bool allowsValueDimension(Index n) {
  if constexpr (S::ValueDimensionStep == 0) {
    return n == S::FixedValueDimension;
  } else {
    return n >= S::FixedValueDimension && (n - S::FixedValueDimension) % S::ValueDimensionStep == 0 &&
           (S::MaxValueDimension == Dynamic || n <= S::MaxValueDimension);
  }
}

namespace internal {

/** \internal Resizes \a space to \a n coordinates if it has resize(n), after checking that \a n is a valid
 * value dimension for its type, and checks that it then has it. A space with several resizable
 * blocks has no resize(n), so it must already have \a n.
 */
template <typename S>
void fitValueDimension(S& space, Index n) {
  eigen_assert(allowsValueDimension<S>(n) && "the dimension does not fit the space");
  if constexpr (requires { space.resize(n); }) {
    if (space.valueDimension() != n) {
      space.resize(n);
    }
    eigen_assert(space.valueDimension() == n && "the space's resize() did not give it this dimension");
  } else {
    eigen_assert(space.valueDimension() == n && "the dimension does not fit the space");
  }
}

}  // namespace internal

template <Space S>
class DualSpace;

}  // namespace Eigen

#endif  // EIGEN_SPACES_SPACE_H
