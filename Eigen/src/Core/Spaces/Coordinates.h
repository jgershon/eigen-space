// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_COORDINATES_H
#define EIGEN_SPACES_COORDINATES_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

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
 * \brief \a Dimension_ plain coordinates, or a resizable run of at most \a MaxDimension_ of them if
 * \a Dimension_ is \c Dynamic.
 *
 * The default space everywhere: when fixed, an empty type. It has no normalize() and is its
 * own difference space, which is what makes it linear (design/decisions.md [linear-space]).
 */
template <int Dimension_ = 1, int MaxDimension_ = Dimension_>
struct Linear : internal::LeafExtent<Dimension_, MaxDimension_> {
  static_assert(Dimension_ == Dynamic || Dimension_ >= 0);
  static_assert(Dimension_ == Dynamic || MaxDimension_ == Dimension_, "a fixed size is its own maximum");
  static constexpr int FixedValueDimension = Dimension_ == Dynamic ? 0 : Dimension_;  ///< the fixed coordinates
  static constexpr int ValueDimensionStep = Dimension_ == Dynamic ? 1 : 0;  ///< one coordinate per element if resizable
  static constexpr int MaxValueDimension = MaxDimension_;                   ///< the largest valid size, or Dynamic

  Linear() = default;

  /** \a n coordinates of a resizable Linear. */
  explicit Linear(Index n)
    requires(Dimension_ == Dynamic)
  {
    this->resize(n);
  }

  /** Equal when every stored parameter and size is; a fixed-size Linear is always equal. */
  friend bool operator==(const Linear&, const Linear&) = default;
};

/** \ingroup Spaces_Module
 * \brief Where a Periodic coordinate's values lie: the half-open interval
 * <tt>[lower, upper)</tt>, whose width is the period. Both ends must be given: an omitted
 * one is NaN, which Periodic rejects.
 */
struct PeriodicInterval {
  double lower = std::numeric_limits<double>::quiet_NaN();  ///< the included lower end
  double upper = std::numeric_limits<double>::quiet_NaN();  ///< the excluded upper end, lower + period
};

/** \ingroup Spaces_Module
 * \brief \a Dimension_ coordinates that wrap with a common period, such as angles or DFT bins.
 *
 * Values are kept in <tt>[lower(), upper())</tt> and differences in
 * <tt>[-period()/2, period()/2)</tt>: the difference space is the Periodic on that interval,
 * so RadiansPi is its own and Degrees360's is Degrees180. The interval is a template argument, so a fixed-size
 * Periodic is an empty type: <tt>struct Bin : Periodic<{.lower = 0.0, .upper = 256.0}> {};</tt>.
 * The interval has no default (design/decisions.md [coordinate-parameters]). The common angles are named
 * RadiansPi, Radians2Pi, Degrees180, and Degrees360.
 *
 * A Gaussian on a Periodic coordinate is a Gaussian in its tangent space, valid while its standard
 * deviation is small against the period; a spread on the order of the period is better held by
 * particles, a grid, or a mixture [tangent-gaussians].
 */
template <PeriodicInterval Interval, int Dimension_ = 1, int MaxDimension_ = Dimension_>
class Periodic : public internal::LeafExtent<Dimension_, MaxDimension_> {
 public:
  static_assert(Dimension_ == Dynamic || Dimension_ >= 1);
  static_assert(Dimension_ == Dynamic || MaxDimension_ == Dimension_, "a fixed size is its own maximum");
  static_assert(Interval.lower == Interval.lower, "the lower end must be given");
  static_assert(Interval.upper == Interval.upper, "the upper end must be given");
  static_assert(Interval.lower < Interval.upper, "the upper end must be above the lower");
  static_assert(Interval.upper - Interval.lower < std::numeric_limits<double>::infinity(), "the period must be finite");

  static constexpr int FixedValueDimension = Dimension_ == Dynamic ? 0 : Dimension_;  ///< the fixed coordinates
  static constexpr int ValueDimensionStep = Dimension_ == Dynamic ? 1 : 0;  ///< one coordinate per element if resizable
  static constexpr int MaxValueDimension = MaxDimension_;                   ///< the largest valid size, or Dynamic

  Periodic() = default;

  /** \a n coordinates of a resizable Periodic. */
  explicit Periodic(Index n)
    requires(Dimension_ == Dynamic)
  {
    this->resize(n);
  }

  /** \returns the lower end of the interval of values */
  static consteval double lower() { return Interval.lower; }

  /** \returns the excluded upper end of the interval of values */
  static consteval double upper() { return Interval.upper; }

  /** \returns the period, <tt>upper() - lower()</tt> */
  static consteval double period() { return Interval.upper - Interval.lower; }

  /** Wraps every value in \a block into <tt>[lower(), upper())</tt>, writing through the const
   * reference (see writable()).
   */
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) const {
    writable(block) = block.unaryExpr([](auto x) { return internal::wrapValue<lower(), period()>(x); });
  }

  /** The Periodic on <tt>[-period()/2, period()/2)</tt>, with the same size. */
  auto difference() const {
    using Difference =
        Periodic<PeriodicInterval{.lower = -period() / 2, .upper = period() / 2}, Dimension_, MaxDimension_>;
    if constexpr (Dimension_ == Dynamic) {
      return Difference{this->valueDimension()};
    } else {
      return Difference{};
    }
  }

  /** Equal when every stored parameter and size is; a fixed-size Periodic is always equal. */
  friend bool operator==(const Periodic&, const Periodic&) = default;
};

/** \ingroup Spaces_Module
 * \brief \a Dimension angles in radians, in <tt>[-π, π)</tt>.
 */
template <int Dimension = 1, int MaxDimension = Dimension>
using RadiansPi =
    Periodic<PeriodicInterval{.lower = -std::numbers::pi, .upper = std::numbers::pi}, Dimension, MaxDimension>;

/** \ingroup Spaces_Module
 * \brief \a Dimension angles in radians, in <tt>[0, 2π)</tt>.
 */
template <int Dimension = 1, int MaxDimension = Dimension>
using Radians2Pi = Periodic<PeriodicInterval{.lower = 0.0, .upper = 2 * std::numbers::pi}, Dimension, MaxDimension>;

/** \ingroup Spaces_Module
 * \brief \a Dimension angles in degrees, in <tt>[-180, 180)</tt>.
 */
template <int Dimension = 1, int MaxDimension = Dimension>
using Degrees180 = Periodic<PeriodicInterval{.lower = -180.0, .upper = 180.0}, Dimension, MaxDimension>;

/** \ingroup Spaces_Module
 * \brief \a Dimension angles in degrees, in <tt>[0, 360)</tt>, such as a compass heading.
 */
template <int Dimension = 1, int MaxDimension = Dimension>
using Degrees360 = Periodic<PeriodicInterval{.lower = 0.0, .upper = 360.0}, Dimension, MaxDimension>;

}  // namespace Eigen

#endif  // EIGEN_SPACES_COORDINATES_H
