// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_LINEAR_H
#define EIGEN_SPACES_LINEAR_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

// QLinear and the compile-time dimensions come before ForwardDeclarations.h, which spells Matrix and Array as
// QMatrix and QArray on QLinear spaces (design/decisions.md [q-names]); nothing here may need a dense type.

namespace Eigen {

/** \ingroup Spaces_Module
 * The value dimension of \a S at compile time, or \c Dynamic if it is resizable.
 */
template <typename S>
inline constexpr int valueDimensionAtCompileTime = S::ValueDimensionStep == 0 ? S::FixedValueDimension : Dynamic;

/** \ingroup Spaces_Module
 * The largest valid value dimension of \a S, or \c Dynamic if it is unbounded.
 */
template <typename S>
inline constexpr int maxValueDimensionAtCompileTime =
    S::ValueDimensionStep == 0 ? S::FixedValueDimension : S::MaxValueDimension;

namespace internal {

/** \internal The value dimension of a coordinate type: a compile-time constant when fixed, which takes no
 * storage.
 */
template <int Dimension, int MaxDimension>
class LeafExtent {
 public:
  /** \returns the number of coordinates, \a Dimension */
  static constexpr Index valueDimension() { return Dimension; }

  /** Always equal: a fixed extent has no state. */
  friend bool operator==(const LeafExtent&, const LeafExtent&) = default;
};

/** \internal The value dimension of a resizable coordinate type: stored, and at most \a MaxDimension. */
template <int MaxDimension>
class LeafExtent<Dynamic, MaxDimension> {
 public:
  /** \returns the current number of coordinates */
  Index valueDimension() const { return m_size; }

  /** Resizes to \a n coordinates. */
  void resize(Index n) {
    eigen_assert(n >= 0 && (MaxDimension == Dynamic || n <= MaxDimension) && "the dimension does not fit the space");
    m_size = n;
  }

  /** Equal when the sizes are. */
  friend bool operator==(const LeafExtent&, const LeafExtent&) = default;

 private:
  Index m_size = 0;  ///< the current number of coordinates
};

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief \a Dimension_ plain coordinates measuring \a Q, or a resizable run of at most \a MaxDimension_ of them if
 * \a Dimension_ is \c Dynamic.
 *
 * The default space everywhere: when fixed, an empty type. It has no normalize() and is its
 * own difference space, which is what makes it linear (design/decisions.md [linear-space]). \a Q is an mp-units
 * quantity spec (<tt>QLinear<isq::displacement, 3></tt>, for a tag, whose unit In binds), unit
 * (<tt>QLinear<si::metre, 3></tt>), or reference (<tt>QLinear<isq::height[si::metre]></tt>); it comes first so that
 * the maximum dimension need not be written to give it, and the dimensionless leaf is spelled Linear
 * [leaf-quantities].
 */
template <internal::QuantityArgument auto Q, int Dimension_ = 1, int MaxDimension_ = Dimension_>
struct QLinear : internal::LeafExtent<Dimension_, MaxDimension_> {
  static_assert(Dimension_ == Dynamic || Dimension_ >= 0);
  static_assert(Dimension_ == Dynamic || MaxDimension_ == Dimension_, "a fixed size is its own maximum");
  static constexpr int FixedValueDimension = Dimension_ == Dynamic ? 0 : Dimension_;  ///< the fixed coordinates
  static constexpr int ValueDimensionStep = Dimension_ == Dynamic ? 1 : 0;  ///< one coordinate per element if resizable
  static constexpr int MaxValueDimension = MaxDimension_;                   ///< the largest valid size, or Dynamic

  QLinear() = default;

  /** \a n coordinates of a resizable QLinear. */
  explicit QLinear(Index n)
    requires(Dimension_ == Dynamic)
  {
    this->resize(n);
  }

  /** \returns what every coordinate measures, \a Q: an mp-units quantity spec, unit, or reference */
  static constexpr auto quantity() { return Q; }

  /** Equal when every stored parameter and size is; a fixed-size QLinear is always equal. */
  friend bool operator==(const QLinear&, const QLinear&) = default;
};

/** \ingroup Spaces_Module
 * \brief \a Dimension dimensionless plain coordinates: QLinear with the quantity left out.
 */
template <int Dimension = 1, int MaxDimension = Dimension>
using Linear = QLinear<dimensionless, Dimension, MaxDimension>;

}  // namespace Eigen

#endif  // EIGEN_SPACES_LINEAR_H
