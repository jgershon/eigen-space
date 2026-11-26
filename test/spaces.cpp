// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The continuous spaces of Core/Spaces (design/decisions.md [sizes-and-dimensions], [no-bounded-coordinates],
// [index-names], [difference-spaces]): sizes, strides, and compile-time bounds of Linear, Periodic, Composite, and
// Repeated; wrapping of values and differences; tag paths and idx(); the Space concept, what satisfies it and what a
// user-written space must provide; and the plus()/minus() operations a space may define.

#include "main.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

namespace {

constexpr double pi = std::numbers::pi;

struct Position : Linear<3> {};
struct Heading : RadiansPi<> {};
struct Range : Linear<1> {};
using State = Composite<Position, Heading, Range>;

struct Ownship : Linear<6> {};
struct Latitude : Linear<1> {};
struct Longitude : RadiansPi<> {};
struct Altitude : Linear<1> {};
struct Vehicle : Composite<Latitude, Longitude, Altitude> {};
struct Vehicles : Repeated<Vehicle> {};
using World = Composite<Ownship, Vehicles>;

// Two groups that both contain an X, so a bare X is ambiguous.
struct X : Linear<1> {};
struct Y : Linear<1> {};
struct Left : Composite<X, Y> {};
struct Right : Composite<X> {};
using Sides = Composite<Left, Right>;

// The default space takes no storage.
struct WithEmptySpace {
  EIGEN_NO_UNIQUE_ADDRESS Linear<4> space;
  double value;
};
static_assert(sizeof(WithEmptySpace) == sizeof(double));

// A composite stores only its blocks with state: fixed-size composites, nested or not, are
// empty, and a resizable one is just its count.
static_assert(std::is_empty_v<State> && std::is_empty_v<Vehicle> && std::is_empty_v<Sides>);
static_assert(std::is_empty_v<Composite<X, X>>);
static_assert(sizeof(World) == sizeof(Index) && sizeof(Vehicles) == sizeof(Index));

VectorXd iota(Index n) { return VectorXd::LinSpaced(n, 0.0, double(n - 1)); }

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

// FixedValueDimension, ValueDimensionStep, and valueDimensionAtCompileTime<> of Linear, a fixed
// Composite, and one with a Repeated block; isLinear<> of each; and allowsValueDimension() accepts
// only the fixed part plus whole repetitions.
void value_dimensions_and_steps() {
  static_assert(Linear<3>::FixedValueDimension == 3 && Linear<3>::ValueDimensionStep == 0);
  static_assert(Linear<Dynamic>::FixedValueDimension == 0 && Linear<Dynamic>::ValueDimensionStep == 1);
  static_assert(State::FixedValueDimension == 5 && State::ValueDimensionStep == 0 &&
                valueDimensionAtCompileTime<State> == 5);
  static_assert(World::FixedValueDimension == 6 && World::ValueDimensionStep == 3 &&
                valueDimensionAtCompileTime<World> == Dynamic);
  static_assert(!isLinear<State> && isLinear<Composite<Position, Altitude>> && isLinear<Repeated<Ownship>>);

  VERIFY(allowsValueDimension<State>(5));
  VERIFY(!allowsValueDimension<State>(6));
  VERIFY(allowsValueDimension<World>(6));
  VERIFY(allowsValueDimension<World>(12));
  VERIFY(!allowsValueDimension<World>(10));
  VERIFY(!allowsValueDimension<World>(3));
}

// MaxValueDimension and maxValueDimensionAtCompileTime<> are Dynamic for an uncapped Repeated and
// the sum of bounds for a capped one or a bounded Linear, and allowsValueDimension() respects the cap.
void maximum_value_dimensions() {
  struct Fleet : Repeated<Vehicle, 4> {};
  using CappedWorld = Composite<Ownship, Fleet>;
  static_assert(World::MaxValueDimension == Dynamic && maxValueDimensionAtCompileTime<World> == Dynamic);
  static_assert(CappedWorld::MaxValueDimension == 6 + 3 * 4 && maxValueDimensionAtCompileTime<CappedWorld> == 18);
  static_assert(maxValueDimensionAtCompileTime<State> == 5);
  static_assert(Linear<Dynamic, 8>::MaxValueDimension == 8);
  VERIFY(allowsValueDimension<CappedWorld>(18));
  VERIFY(!allowsValueDimension<CappedWorld>(21));
  VERIFY((allowsValueDimension<Linear<Dynamic, 8>>(8)));
  VERIFY(!(allowsValueDimension<Linear<Dynamic, 8>>(9)));
}

// A periodic space wraps values into [lower, upper) and differences into the centered period, for
// RadiansPi and an explicit [0, 256) period.
void periodic_wraps_values_and_differences() {
  const auto angles = RadiansPi<3>{};
  auto values = Vector3d{3.5, -3.5, 7.0};
  normalizeValues(angles, values);
  VERIFY(near(values(0), 3.5 - 2 * pi, 1e-15));
  VERIFY(near(values(1), -3.5 + 2 * pi, 1e-15));
  VERIFY(near(values(2), 7.0 - 2 * pi, 1e-15));

  auto differences = Vector3d{6.0, -6.0, 0.5};
  normalizeDifferences(angles, differences);
  VERIFY(near(differences(0), 6.0 - 2 * pi, 1e-15));
  VERIFY(near(differences(1), -6.0 + 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(differences(2), 0.5);

  const auto bins = Periodic<{.lower = 0.0, .upper = 256.0}>{};
  auto bin = Matrix<double, 1, 3>{-1.0, 256.0, 300.0};
  normalizeValues(bins, bin);
  VERIFY_IS_EQUAL(bin, (Matrix<double, 1, 3>{255.0, 0.0, 44.0}));
}

// Values many periods away wrap correctly, a value that rounds onto the upper bound stays inside
// the half-open interval, and NaN and infinity become NaN.
void periodic_wraps_values_far_outside() {
  const auto bins = Periodic<{.lower = 0.0, .upper = 256.0}>{};
  auto far = Vector4d{1000.5, -1000.5, 512.0, -512.0};
  normalizeValues(bins, far);
  VERIFY_IS_EQUAL(far, Vector4d(232.5, 23.5, 0.0, 0.0));

  // -1e-20 + 360 rounds to 360, which the half-open interval excludes.
  auto edge = Vector2d{-1e-20, 360.0 + 1e-13};
  normalizeValues(Degrees360<2>{}, edge);
  VERIFY_IS_EQUAL(edge(0), 0.0);
  VERIFY(edge(1) >= 0.0);
  VERIFY(edge(1) < 360.0);

  auto invalid = Vector2d{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
  normalizeValues(Degrees360<2>{}, invalid);
  VERIFY((std::isnan)(invalid(0)));
  VERIFY((std::isnan)(invalid(1)));
}

// Values within rounding of a seam far outside the interval wrap into it. Far out, the quotient in
// x - p floor((x - a) / p) can round up to the next integer, which put 9.5% of the doubles nearest
// the odd multiples of pi below -pi, by up to 1.1e-10 [coordinate-parameters]. These are the three doubles
// nearest each of the first 200,000 odd multiples of pi on each side.
void periodic_wraps_values_near_far_seams() {
  constexpr int count = 200'000;
  auto values = VectorXd(6 * count);
  for (int k = 0; k < count; ++k) {
    const double seam = (2 * k + 1) * pi;
    for (int sign : {-1, 1}) {
      const Index i = 3 * (2 * k + (sign + 1) / 2);
      values(i) = std::nextafter(sign * seam, -std::numeric_limits<double>::infinity());
      values(i + 1) = sign * seam;
      values(i + 2) = std::nextafter(sign * seam, std::numeric_limits<double>::infinity());
    }
  }
  normalizeValues(RadiansPi<Dynamic>{values.size()}, values);
  VERIFY(values.minCoeff() >= -pi);
  VERIFY(values.maxCoeff() < pi);
}

// The named angle spaces have the expected lower bounds and periods, Degrees360 is the Periodic of
// its interval, all are empty types, and each wraps as named.
void named_angles() {
  static_assert(RadiansPi<>::lower() == -pi && RadiansPi<>::period() == 2 * pi);
  static_assert(Radians2Pi<>::lower() == 0.0 && Radians2Pi<>::period() == 2 * pi);
  static_assert(Degrees180<>::lower() == -180.0 && Degrees360<>::lower() == 0.0 && Degrees360<>::period() == 360.0);
  static_assert(std::is_same_v<Periodic<{.lower = 0.0, .upper = 360.0}>, Degrees360<>>);
  static_assert(std::is_empty_v<Degrees360<3>> && std::is_empty_v<RadiansPi<2>>);

  auto centered = Vector4d{-1.0, 361.0, 180.0, -180.0};
  normalizeValues(Degrees180<4>{}, centered);
  VERIFY_IS_EQUAL(centered, Vector4d(-1.0, 1.0, -180.0, -180.0));

  auto compass = Vector4d{-1.0, 361.0, 180.0, -180.0};
  normalizeValues(Degrees360<4>{}, compass);
  VERIFY_IS_EQUAL(compass, Vector4d(359.0, 1.0, 180.0, 180.0));

  auto radians = Vector2d{-1.0, 7.0};
  normalizeValues(Radians2Pi<2>{}, radians);
  VERIFY(near(radians(0), 2 * pi - 1.0, 1e-15));
  VERIFY(near(radians(1), 7.0 - 2 * pi, 1e-15));
}

// normalizeValues() on a Composite applies each block's rule to its rows of every column of a
// sample matrix.
void composite_normalizes_each_block_and_every_sample() {
  const auto state = State{};
  auto samples = Matrix<double, 5, 2>{};
  samples.col(0) << 10.0, 20.0, 30.0, 4.0, -1.0;
  samples.col(1) << -10.0, 0.0, 0.0, -4.0, 2.0;
  normalizeValues(state, samples);
  VERIFY_IS_EQUAL(samples.col(0).head<3>(), Vector3d(10.0, 20.0, 30.0));
  VERIFY(near(samples(3, 0), 4.0 - 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(samples(4, 0), -1.0);
  VERIFY(near(samples(3, 1), -4.0 + 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(samples(4, 1), 2.0);
}

template <typename S>
concept MinusWithoutInstance = requires(const VectorXd& x) { minus<S>(x, x); };

// A fixed-size space, given as a template argument, does what its default instance does: plus(),
// minus(), and both normalizations, wrapping the heading. A resizable space holds its count, so it
// is not one and must be passed.
void a_fixed_size_space_needs_no_instance() {
  static_assert(FixedSizeSpace<State> && FixedSizeSpace<Sides> && FixedSizeSpace<Linear<4>>);
  static_assert(!FixedSizeSpace<World> && !FixedSizeSpace<Vehicles>);
  static_assert(MinusWithoutInstance<State> && !MinusWithoutInstance<World>);
  const auto x = Vector<double, 5>{1.0, 2.0, 3.0, 3.0, -1.0};
  const auto d = Vector<double, 5>{0.5, 0.0, 0.0, 1.0, 2.0};
  VERIFY_IS_EQUAL(plus<State>(x, d), plus(State{}, x, d));
  VERIFY_IS_EQUAL(minus<State>(d, x), minus(State{}, d, x));
  VERIFY(near(plus<State>(x, d)(3), 4.0 - 2 * pi, 1e-15));

  Vector<double, 5> values = Vector<double, 5>::Constant(4.0);
  Vector<double, 5> differences = values;
  normalizeValues<State>(values);
  normalizeDifferences<State>(differences);
  VERIFY(near(values(3), 4.0 - 2 * pi, 1e-15));
  VERIFY(near(differences(3), 4.0 - 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(values(4), 4.0);
}

// A Repeated block wraps the periodic coordinate of every repetition and leaves the linear ones alone.
void repeated_normalizes_every_repetition() {
  auto world = World{};
  world.resize(12);  // two vehicles
  VERIFY_IS_EQUAL(world.block<1>().count(), 2);
  VectorXd x = VectorXd::Zero(12);
  x(7) = 4.0;    // vehicle 0 longitude
  x(10) = -4.0;  // vehicle 1 longitude
  x(8) = 4.0;    // vehicle 0 altitude, linear
  normalizeValues(world, x);
  VERIFY(near(x(7), 4.0 - 2 * pi, 1e-15));
  VERIFY(near(x(10), -4.0 + 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(x(8), 4.0);
}

// idx<Space, Tag>() is a constexpr Index for a fixed space, and indexing a vector with a
// block's idx gives a fixed-size segment.
void index_fixed_tags() {
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(idx<State, Heading>())>, Index>);
  static_assert(idx<State, Heading>() == 3);  // usable in constant expressions
  static_assert(idx<State, Range>() == 4);

  const Matrix<double, 5, 1> x = iota(5);
  const auto position = x(idx<State, Position>());
  static_assert(decltype(position)::RowsAtCompileTime == 3);
  VERIFY_IS_EQUAL(Vector3d{position}, Vector3d(0.0, 1.0, 2.0));
}

// idx<S, Path>() through a Repeated block selects every repetition's coordinates, and
// idx<S, Path>(k) selects one repetition or one coordinate of it, as diffIdx<S, Path>(k) does of
// the differences, which are the values on these linear blocks.
void index_repeated_tail() {
  const VectorXd x = iota(12);
  VERIFY_IS_EQUAL(VectorXd{x(idx<World, Ownship>())}, iota(6));
  VERIFY_IS_EQUAL(VectorXd{x(idx<World, Vehicles>())}, (iota(6).array() + 6.0).matrix());
  VERIFY_IS_EQUAL(VectorXd{x(idx<World, TagPath<Vehicles, Altitude>>())}, Vector2d(8.0, 11.0));
  VERIFY_IS_EQUAL(VectorXd{x(idx<World, TagPath<Vehicles, Longitude>>())}, Vector2d(7.0, 10.0));

  VERIFY_IS_EQUAL(Vector3d{x(idx<World, Vehicles>(1))}, Vector3d(9.0, 10.0, 11.0));
  VERIFY_IS_EQUAL((idx<World, TagPath<Vehicles, Longitude>>(1)), 10);
  VERIFY_IS_EQUAL((idx<World, TagPath<Vehicles, Altitude>>(0)), 8);
  VERIFY_IS_EQUAL((diffIdx<World, TagPath<Vehicles, Longitude>>(1)), 10);

  // From a space's run-time sizes: every path, with the repetition count read from the space.
  auto world = World{};
  world.resize(12);
  VERIFY_IS_EQUAL(VectorXd{x(idx<Vehicles>(world))}, (iota(6).array() + 6.0).matrix());
  VERIFY_IS_EQUAL(VectorXd{x(idx<TagPath<Vehicles, Altitude>>(world))}, Vector2d(8.0, 11.0));
  VERIFY_IS_EQUAL(Vector3d{x(idx<Vehicles>(world, 1))}, Vector3d(9.0, 10.0, 11.0));
  VERIFY_IS_EQUAL(idx<Heading>(State{}), 3);
  const BlockRows rows = blockRows<Vehicles>(world);
  VERIFY_IS_EQUAL(rows.start, 6);
  VERIFY_IS_EQUAL(rows.size, 6);
}

// A tag that appears in two groups is addressed by a TagPath through its group; a unique tag needs
// no qualification.
void index_qualified_paths() {
  static_assert(idx<Sides, TagPath<Left, X>>() == 0);
  static_assert(idx<Sides, TagPath<Right, X>>() == 2);
  static_assert(idx<Sides, Y>() == 1);  // unique, so no qualification needed
}

// The internal complementSubspace() removes a block or a nested path and keeps the rest, replacing
// a composite that loses a block by what remains and keeping a Repeated block's count when one
// coordinate of every repetition goes.
void complement_keeps_the_remaining_blocks() {
  using internal::complementSubspace;
  static_assert(std::is_same_v<decltype(complementSubspace<Heading>(State{})), Composite<Position, Range>>);
  static_assert(std::is_same_v<decltype(complementSubspace<Range>(State{})), Composite<Position, Heading>>);
  static_assert(std::is_same_v<decltype(complementSubspace<Sides, Left, X>(Sides{})), Composite<Y, Right>>);

  auto world = World{};
  world.resize(12);
  const auto rest = complementSubspace<Vehicles, Longitude>(world);
  static_assert(std::is_same_v<decltype(rest), const Composite<Ownship, Repeated<Composite<Latitude, Altitude>>>>);
  VERIFY_IS_EQUAL(rest.block<1>().count(), 2);
  VERIFY_IS_EQUAL(valueDimension(rest), 10);
  const auto ownship = complementSubspace<Vehicles>(world);
  static_assert(std::is_same_v<decltype(ownship), const Ownship>);
}

// Spaces written from scratch, using none of the library's space types: degrees in [0, 360),
// whose differences are degrees in [-180, 180).
struct UserDegreeDifferences {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr Index valueDimension() { return 1; }

  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) const {
    writable(block) = block.unaryExpr([](auto d) { return d - 360.0 * numext::floor(d / 360.0 + 0.5); });
  }

  UserDegreeDifferences difference() const { return *this; }

  friend bool operator==(const UserDegreeDifferences&, const UserDegreeDifferences&) = default;
};

struct UserDegrees {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr Index valueDimension() { return 1; }

  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) const {
    writable(block) = block.unaryExpr([](auto x) { return x - 360.0 * numext::floor(x / 360.0); });
  }

  UserDegreeDifferences difference() const { return {}; }

  friend bool operator==(const UserDegrees&, const UserDegrees&) = default;
};

// Each of these breaks one requirement.
struct NormalizeWithoutDifference : UserDegrees {
  void difference() const = delete;
};

struct DifferenceOfAnotherSize : UserDegrees {
  Linear<2> difference() const { return {}; }
};

struct ResizableWithoutResize : UserDegrees {
  static constexpr int FixedValueDimension = 0;
  static constexpr int ValueDimensionStep = 1;
  static constexpr int MaxValueDimension = Dynamic;
};

struct MaxDimensionOfAnotherSpace : UserDegrees {
  static constexpr int MaxValueDimension = 3;
};

struct NonConstantDimension : UserDegrees {
  static inline int FixedValueDimension = 1;
};

// Operations written for one vector rather than for a block of columns, which the library would
// otherwise skip as missing.
using Vector1 = Matrix<double, 1, 1>;

struct VectorNormalize : UserDegrees {
  void normalize(Vector1& x) const { x(0) -= 360.0 * numext::floor(x(0) / 360.0); }
};

struct VectorPlus : UserDegrees {
  void plus(const Vector1& x, const Vector1& d, Vector1& out) const { out = x + d; }
};

struct VectorMinus : UserDegrees {
  void minus(const Vector1& a, const Vector1& b, Vector1& out) const { out = a - b; }
};

struct VectorNormalizedDifferences : UserDegreeDifferences {
  void normalize(Vector1& d) const { d(0) -= 360.0 * numext::floor(d(0) / 360.0 + 0.5); }
};

struct DifferencesWithVectorNormalize : UserDegrees {
  VectorNormalizedDifferences difference() const { return {}; }
};

// A linear space written from scratch: no normalize() and no difference().
struct UserMeters {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr Index valueDimension() { return 1; }
  friend bool operator==(const UserMeters&, const UserMeters&) = default;
};

// A tag on Linear that wraps like UserDegrees: its own operations make it non-linear.
struct WrappedOffset : Linear<1> {
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) const {
    UserDegrees{}.normalize(block);
  }
  UserDegreeDifferences difference() const { return {}; }
};

// A heading kept unwrapped, so that it plots without jumps, whose residuals still take the short
// way round: plain values, wrapped differences.
struct UnwrappedHeading : Linear<1> {
  RadiansPi<> difference() const { return {}; }
};

// Positive scales whose differences are log ratios: x ⊞ δ = x e^δ and a ⊟ b = log(a / b), which
// no normalize() of a sum or a difference can express.
struct Scale {
  static constexpr int FixedValueDimension = 1;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 1;
  static constexpr Index valueDimension() { return 1; }
  Linear<1> difference() const { return {}; }

  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    writable(out) = (x.array() * d.array().exp()).matrix();
  }

  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    writable(out) = (a.array() / b.array()).log().matrix();
  }

  friend bool operator==(const Scale&, const Scale&) = default;
};

struct Gain : Scale {};

struct Gains : Repeated<Gain> {
  using Repeated::Repeated;
};

// Its own plus() without saying where its differences live.
struct PlusWithoutDifference : Linear<1> {
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    Scale{}.plus(x, d, out);
  }
};

// Near misses of the forms: templates the library would have taken as missing, leaving the space
// a plain vector. A normalize() that is not const, a plus() whose output is a non-const reference,
// and a minus() of the same kind; each is otherwise a working space.
struct NonConstNormalize : UserDegrees {
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& block) {
    UserDegrees{}.normalize(block);
  }
};

struct NonConstOutputPlus : Scale {
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, MatrixBase<Out>& out) const {
    Scale{}.plus(x, d, out);
  }
};

struct NonConstOutputMinus : Scale {
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, MatrixBase<Out>& out) const {
    Scale{}.minus(a, b, out);
  }
};

// Its own plus() and a difference space, but residuals left to subtraction.
struct PlusWithoutMinus : Linear<1> {
  Linear<1> difference() const { return {}; }

  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    writable(out) = (x.array() * d.array().exp()).matrix();
  }
};

// isLinear<> and its parts take only a Space.
template <typename S>
concept HasLinearity =
    requires { isLinear<S>; } || requires { hasPlainValues<S>; } || requires { hasPlainDifferences<S>; };

// Every library space type satisfies the Space concept, and scalars, vectors, and spaces that break one requirement
// each (no difference space, a difference of another size, resizable without resize(), an inconsistent
// MaxValueDimension, a non-constant FixedValueDimension, a normalize(), plus(), or minus() written for one vector, on
// the space or on its difference space, a near miss of their forms, a plus() without a minus()) are rejected, and
// isLinear<> does not compile for what is not a space.
void concept_accepts_spaces_and_rejects_the_rest() {
  static_assert(Space<Linear<3>> && Space<Linear<Dynamic>> && Space<Linear<Dynamic, 8>>);
  static_assert(Space<Degrees360<2>> && Space<RadiansPi<Dynamic, 4>> && Space<Heading>);
  static_assert(Space<State> && Space<World> && Space<Vehicles> && Space<Sides>);

  static_assert(!Space<double> && !Space<Vector3d>);
  static_assert(!Space<NormalizeWithoutDifference>);
  static_assert(!Space<DifferenceOfAnotherSize>);
  static_assert(!Space<ResizableWithoutResize>);
  static_assert(!Space<MaxDimensionOfAnotherSpace>);
  static_assert(!Space<NonConstantDimension>);
  static_assert(!Space<VectorNormalize> && !Space<VectorPlus> && !Space<VectorMinus>);
  static_assert(!Space<DifferencesWithVectorNormalize>);
  static_assert(!Space<NonConstNormalize> && !Space<NonConstOutputPlus> && !Space<NonConstOutputMinus>);
  static_assert(!Space<PlusWithoutMinus>);
  static_assert(Space<Scale>, "the space the near misses of plus() and minus() start from is one");
  static_assert(HasLinearity<Linear<2>> && !HasLinearity<double> && !HasLinearity<Vector3d>);
  static_assert(Space<UserDegrees>, "the space each of them breaks is one");
}

// isLinear<> is true exactly when a space has no normalize() and no difference(), including for
// spaces written from scratch; a Linear tag with its own operations is non-linear and wraps; on a
// linear space the free normalizers do nothing.
void linearity_comes_from_the_normalize_operations() {
  static_assert(isLinear<Linear<3>> && !isLinear<Degrees360<>> && isLinear<Range> && !isLinear<UserDegrees>);
  static_assert(Space<UserMeters> && isLinear<UserMeters>);
  using Mixed = Composite<Position, WrappedOffset>;
  static_assert(Space<WrappedOffset> && !isLinear<WrappedOffset> && !isLinear<Mixed>);

  auto x = Vector4d{1.0, 2.0, 3.0, 370.0};
  normalizeValues(Mixed{}, x);
  VERIFY_IS_EQUAL(x, Vector4d(1.0, 2.0, 3.0, 10.0));

  auto y = Vector3d{-7.0, 400.0, 1e9};
  normalizeValues(Position{}, y);
  normalizeDifferences(Position{}, y);
  VERIFY_IS_EQUAL(y, Vector3d(-7.0, 400.0, 1e9));
}

// The internal leafSupport() reports the whole line for a linear leaf, the interval and periodicity
// for a Periodic tag, and nothing for a non-linear space written from scratch.
void leaves_report_where_their_values_lie() {
  using internal::leafSupport;
  constexpr double inf = std::numeric_limits<double>::infinity();
  static_assert(leafSupport<Position>()->lower == -inf && leafSupport<Position>()->upper == inf &&
                !leafSupport<Position>()->periodic);
  static_assert(leafSupport<UserMeters>().has_value());
  static_assert(leafSupport<Heading>()->lower == -pi && leafSupport<Heading>()->upper == pi &&
                leafSupport<Heading>()->periodic);
  static_assert(!leafSupport<UserDegrees>().has_value());
}

// hasPlainValues<> and hasPlainDifferences<> are independent: an unwrapped heading keeps its values
// plain and wraps its differences, and a Composite applies each block's rule to values and to
// differences separately.
void values_and_differences_normalize_separately() {
  static_assert(hasPlainValues<UnwrappedHeading> && !hasPlainDifferences<UnwrappedHeading>);
  static_assert(Space<UnwrappedHeading> && !isLinear<UnwrappedHeading>);

  auto heading = Vector2d{7.0, -7.0};
  normalizeValues(UnwrappedHeading{}, heading);
  VERIFY_IS_EQUAL(heading, Vector2d(7.0, -7.0));
  normalizeDifferences(UnwrappedHeading{}, heading);
  VERIFY(near(heading(0), 7.0 - 2 * pi, 1e-15));
  VERIFY(near(heading(1), -7.0 + 2 * pi, 1e-15));

  using Track = Composite<Heading, UnwrappedHeading>;
  static_assert(!hasPlainValues<Track> && !hasPlainDifferences<Track>);
  auto track = Vector2d{-4.0, 7.0};
  normalizeValues(Track{}, track);
  VERIFY(near(track(0), -4.0 + 2 * pi, 1e-15));
  VERIFY_IS_EQUAL(track(1), 7.0);
  auto residual = Vector2d{-4.0, 7.0};
  normalizeDifferences(Track{}, residual);
  VERIFY(near(residual(0), -4.0 + 2 * pi, 1e-15));
  VERIFY(near(residual(1), 7.0 - 2 * pi, 1e-15));
}

// DifferenceSpace<> is the centered period for a periodic space, the space itself for Linear, a
// space's own difference() for a user space, and a Composite of the blocks' difference spaces, a
// tagged block's as DifferenceOf of its tag; a resizable space's difference space carries its sizes, and
// normalizing differences is normalizing values in it.
void differences_live_in_a_difference_space() {
  static_assert(std::is_same_v<DifferenceSpace<Degrees360<2>>, Degrees180<2>>);
  static_assert(std::is_same_v<DifferenceSpace<Radians2Pi<>>, RadiansPi<>>);
  static_assert(std::is_same_v<DifferenceSpace<RadiansPi<>>, RadiansPi<>>);
  static_assert(std::is_same_v<DifferenceSpace<Position>, Position>);
  static_assert(std::is_same_v<DifferenceSpace<UserDegrees>, UserDegreeDifferences>);
  static_assert(std::is_same_v<DifferenceSpace<State>, Composite<Position, DifferenceOf<Heading>, Range>>);
  static_assert(std::is_base_of_v<RadiansPi<>, DifferenceOf<Heading>> &&
                std::is_same_v<DifferenceSpace<DifferenceOf<Heading>>, DifferenceOf<Heading>>);
  static_assert(std::is_same_v<DifferenceSpace<Composite<Position, Altitude>>, Composite<Position, Altitude>>);
  static_assert(std::is_empty_v<DifferenceSpace<State>>);

  const auto bins = differenceSpace(Degrees360<Dynamic>{4});
  static_assert(std::is_same_v<decltype(bins), const Degrees180<Dynamic>>);
  VERIFY_IS_EQUAL(valueDimension(bins), 4);
  auto world = World{};
  world.resize(12);
  const auto worldDifferences = differenceSpace(world);
  VERIFY_IS_EQUAL(valueDimension(worldDifferences), 12);
  VERIFY_IS_EQUAL(worldDifferences.block<1>().count(), 2);

  auto d = Vector2d{190.0, -190.0};
  normalizeDifferences(Degrees360<2>{}, d);
  VERIFY_IS_EQUAL(d, Vector2d(-170.0, 170.0));
  auto v = Vector2d{190.0, -190.0};
  normalizeValues(Degrees180<2>{}, v);
  VERIFY_IS_EQUAL(v, d);
}

// The free plus() and minus() are the normalized sum and difference, broadcast a one-column operand
// across the other's columns, and are plain arithmetic with fixed sizes on a linear space.
void defaults_are_the_normalized_sum_and_difference() {
  const auto bearings = Degrees360<2>{};
  const auto x = Vector2d{350.0, 10.0};
  VERIFY_IS_EQUAL(plus(bearings, x, Vector2d{20.0, -20.0}), Vector2d(10.0, 350.0));
  VERIFY_IS_EQUAL(minus(bearings, Vector2d{10.0, 350.0}, x), Vector2d(20.0, -20.0));

  const auto samples = Matrix<double, 2, 3>{{10.0, 180.0, 355.0}, {0.0, 0.0, 0.0}};
  const Matrix<double, 2, 3> deviations = minus(bearings, samples, x);
  VERIFY_IS_EQUAL(deviations, (Matrix<double, 2, 3>{{20.0, -170.0, 5.0}, {-10.0, -10.0, -10.0}}));
  VERIFY_IS_EQUAL((Matrix<double, 2, 3>{plus(bearings, x, deviations)}), samples);

  const auto sum = plus(Linear<2>{}, x, Vector2d{20.0, -20.0});
  static_assert(std::is_same_v<decltype(sum), const Vector2d>);
  VERIFY_IS_EQUAL(sum, Vector2d(370.0, -10.0));
}

// A space with its own plus() and minus() (a log-ratio scale) satisfies Space only with a
// difference(), is non-linear, and a Composite or Repeated passes each block to its own operations.
void a_space_may_define_its_own_plus_and_minus() {
  static_assert(Space<Scale> && Space<Gain> && Space<Gains> && !Space<PlusWithoutDifference>);
  static_assert(!hasPlainValues<Scale> && !hasPlainDifferences<Scale> && !isLinear<Scale>);
  const auto two = Matrix<double, 1, 1>{2.0};
  VERIFY(near(plus(Gain{}, two, Matrix<double, 1, 1>{std::log(3.0)})(0), 6.0, 1e-14));
  VERIFY(near(minus(Gain{}, Matrix<double, 1, 1>{6.0}, two)(0), std::log(3.0), 1e-15));

  using Mixed = Composite<X, Gain, Heading>;
  static_assert(Space<Mixed> && !isLinear<Mixed>);
  const auto x = Vector3d{1.0, 2.0, 3.0};
  const Vector3d moved = plus(Mixed{}, x, Vector3d{0.5, std::log(2.0), 0.5});
  VERIFY(near(moved(0), 1.5, 1e-15));
  VERIFY(near(moved(1), 4.0, 1e-14));
  VERIFY(near(moved(2), 3.5 - 2 * pi, 1e-15));
  const Vector3d back = minus(Mixed{}, moved, x);
  VERIFY((back - Vector3d{0.5, std::log(2.0), 0.5}).norm() < 1e-14);

  const auto gains = Gains{Gain{}, 3};
  const VectorXd doubled = plus(gains, Vector3d{1.0, 2.0, 4.0}, Vector3d::Constant(std::log(2.0)));
  VERIFY((doubled - Vector3d{2.0, 4.0, 8.0}).norm() < 1e-14);
}

namespace adl_trap {

// A space whose namespace has free plus() and minus() that take anything: an unqualified call
// inside the library would find them by argument-dependent lookup and be ambiguous with
// Eigen::plus() and Eigen::minus(), or call them.
struct Meters : Linear<2> {};
struct Turn : RadiansPi<> {};
using Track = Composite<Meters, Turn>;

template <typename... Args>
int plus(const Track&, const Args&...) {
  return 0;
}

template <typename... Args>
int minus(const Track&, const Args&...) {
  return 0;
}

}  // namespace adl_trap

// The library calls plus() and minus() qualified, so a free plus() or minus() in a space's namespace
// is never found: the composite's own operations wrap the heading. Customization is by members only.
void free_functions_in_the_spaces_namespace_are_not_found() {
  using adl_trap::Track;
  const auto x = Vector3d{1.0, 2.0, 3.0};
  const Vector3d moved = Eigen::plus(Track{}, x, Vector3d{0.0, 0.0, 1.0});
  VERIFY(near(moved(2), 4.0 - 2 * pi, 1e-15));
  const Vector3d back = Eigen::minus(Track{}, moved, x);
  VERIFY(near(back(2), 1.0, 1e-15));
  static_assert(std::is_same_v<decltype(Eigen::plus<Track>(x, x)), Vector3d>);
}

}  // namespace

EIGEN_DECLARE_TEST(spaces) {
  CALL_SUBTEST_1(value_dimensions_and_steps());
  CALL_SUBTEST_1(maximum_value_dimensions());
  CALL_SUBTEST_1(periodic_wraps_values_and_differences());
  CALL_SUBTEST_1(periodic_wraps_values_far_outside());
  CALL_SUBTEST_1(periodic_wraps_values_near_far_seams());
  CALL_SUBTEST_1(named_angles());
  CALL_SUBTEST_1(composite_normalizes_each_block_and_every_sample());
  CALL_SUBTEST_1(a_fixed_size_space_needs_no_instance());
  CALL_SUBTEST_1(repeated_normalizes_every_repetition());
  CALL_SUBTEST_2(index_fixed_tags());
  CALL_SUBTEST_2(index_repeated_tail());
  CALL_SUBTEST_2(index_qualified_paths());
  CALL_SUBTEST_2(complement_keeps_the_remaining_blocks());
  CALL_SUBTEST_3(concept_accepts_spaces_and_rejects_the_rest());
  CALL_SUBTEST_3(linearity_comes_from_the_normalize_operations());
  CALL_SUBTEST_3(leaves_report_where_their_values_lie());
  CALL_SUBTEST_3(values_and_differences_normalize_separately());
  CALL_SUBTEST_3(differences_live_in_a_difference_space());
  CALL_SUBTEST_4(defaults_are_the_normalized_sum_and_difference());
  CALL_SUBTEST_4(a_space_may_define_its_own_plus_and_minus());
  CALL_SUBTEST_4(free_functions_in_the_spaces_namespace_are_not_found());
}
