// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The unit on every leaf space and the dual (design/decisions.md [mandatory-units], [dual-spaces]): the Unit
// concept and Eigen's dimensionless identity, QLinear and QPeriodic with the plain names as their dimensionless
// aliases, unitAt() and unitOf() through composites, repeated blocks, and duals, and Dual's collapse rules. A unit
// family of integer exponents stands in for a units library; with EIGEN_TEST_MPUNITS the same checks run again on
// mp-units' units.

// Before main.h, whose forbidden-identifier macros (I, min, max) would break mp-units' headers.
#ifdef EIGEN_TEST_MPUNITS
#include <mp-units/systems/si.h>
#include <mp-units/systems/angular.h>
#endif

#include "main.h"

#include <numbers>
#include <type_traits>

namespace {

// A unit as a vector of integer exponents over length, time, and angle: closed under product and quotient, with
// constexpr equality, and structural, so that it can be a template argument. Dimensionless is the zero vector.
struct TestUnit {
  int length = 0;
  int time = 0;
  int angle = 0;

  friend constexpr TestUnit operator*(TestUnit a, TestUnit b) {
    return {a.length + b.length, a.time + b.time, a.angle + b.angle};
  }
  friend constexpr TestUnit operator/(TestUnit a, TestUnit b) {
    return {a.length - b.length, a.time - b.time, a.angle - b.angle};
  }
  friend constexpr bool operator==(TestUnit, TestUnit) = default;
};

constexpr TestUnit one{};
constexpr TestUnit metre{1, 0, 0};
constexpr TestUnit second{0, 1, 0};
constexpr TestUnit radian{0, 0, 1};
constexpr TestUnit metrePerSecond = metre / second;

struct Position : QLinear<metre, 3> {};
struct Velocity : QLinear<metrePerSecond, 3> {};
struct Heading : QRadiansPi<radian> {};
struct Landmarks : QLinear<metre, Dynamic, 64> {};
struct Range : Linear<1> {};
using State = Composite<Position, Velocity, Heading>;
using Map = Composite<Position, Landmarks>;

struct Vehicle : Composite<Position, Heading> {};
struct Fleet : Repeated<Vehicle, 4> {
  using Repeated::Repeated;
};

// A space written from scratch declares no unit, so it is dimensionless.
struct FromScratch {
  static constexpr int FixedValueDimension = 2;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 2;
  static constexpr Index valueDimension() { return 2; }
  friend bool operator==(const FromScratch&, const FromScratch&) = default;
};

// The Unit concept accepts unit-like values and the dimensionless identity, and rejects numbers, which would
// otherwise read as the unit of a leaf given its dimension in the wrong slot.
void concept_accepts_units_and_rejects_numbers() {
  static_assert(Unit<TestUnit> && Unit<Dimensionless>);
  static_assert(!Unit<int> && !Unit<double> && !Unit<Vector3d>);
}

// Eigen's dimensionless unit is the identity of every unit algebra: a foreign unit absorbs it on either side of a
// product or quotient, dividing it by a unit inverts the unit, and a unit equals it exactly when the unit is the
// dimensionless one of its own system.
void dimensionless_is_the_identity() {
  static_assert(metre * dimensionless == metre && dimensionless * metre == metre);
  static_assert(metre / dimensionless == metre);
  static_assert(dimensionless / metre == one / metre);
  static_assert(dimensionless == dimensionless);
  static_assert(one == dimensionless && dimensionless == one);
  static_assert(!(metre == dimensionless) && !(dimensionless == metre));
  static_assert(internal::isDimensionless(one) && internal::isDimensionless(dimensionless));
  static_assert(!internal::isDimensionless(metre));
  static_assert(internal::unitsEqual(one, dimensionless) && internal::unitsEqual(metre, metre));
  static_assert(!internal::unitsEqual(metre, second));
  static_assert(internal::inverseUnit(metre) == one / metre);
  static_assert(internal::inverseUnit(dimensionless) == dimensionless);
}

// The plain names are the dimensionless leaves: Linear<N> is QLinear<dimensionless, N>, with the same for Periodic and
// the named angles, so every spelling written before the unit existed keeps its type.
void plain_names_are_the_dimensionless_aliases() {
  static_assert(std::is_same_v<Linear<3>, QLinear<dimensionless, 3>>);
  static_assert(std::is_same_v<Linear<Dynamic, 8>, QLinear<dimensionless, Dynamic, 8>>);
  static_assert(std::is_same_v<RadiansPi<2>, QRadiansPi<dimensionless, 2>>);
  static_assert(std::is_same_v<Degrees360<>, QDegrees360<dimensionless>>);
  static_assert(std::is_same_v<Periodic<{.lower = 0.0, .upper = 256.0}>,
                               QPeriodic<dimensionless, {.lower = 0.0, .upper = 256.0}>>);
  static_assert(Linear<3>::unit() == dimensionless && RadiansPi<>::unit() == dimensionless);
  static_assert(Position::unit() == metre && Heading::unit() == radian && Landmarks::unit() == metre);
  static_assert(std::is_empty_v<Position> && std::is_empty_v<State>);
}

// A unit-carrying leaf is a Space like its dimensionless alias: it wraps, differs, and resizes the same way, and its
// difference space keeps the unit.
void unit_leaves_are_spaces() {
  static_assert(Space<Position> && Space<Heading> && Space<Landmarks> && Space<State> && Space<Map>);
  static_assert(isLinear<Position> && !isLinear<Heading> && !isLinear<State>);
  static_assert(std::is_same_v<DifferenceSpace<Heading>, QRadiansPi<radian>>);
  static_assert(std::is_same_v<DifferenceSpace<QDegrees360<radian>>, QDegrees180<radian>>);
  static_assert(DifferenceSpace<Heading>::unit() == radian);

  auto x = Vector<double, 7>{1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0};
  normalizeValues(State{}, x);
  VERIFY(std::abs(x(6) - (4.0 - 2 * std::numbers::pi)) <= 1e-15);
  auto map = Map{};
  map.resize(9);
  VERIFY_IS_EQUAL(valueDimension(map), 9);
  VERIFY_IS_EQUAL(map.block<1>().valueDimension(), 6);
}

// unitAt() reads a block's unit by tag path, through composites and repeated blocks; unitOf() reads a whole space's
// where its blocks agree; a space written from scratch is dimensionless.
void unit_at_a_path() {
  static_assert(unitAt<State, Position>() == metre);
  static_assert(unitAt<State, Velocity>() == metre / second);
  static_assert(unitAt<State, Heading>() == radian);
  static_assert(unitAt<Map, Landmarks>() == metre);
  using Convoy = Composite<Fleet, Range>;
  static_assert(unitAt<Convoy, TagPath<Fleet, Heading>>() == radian);
  static_assert(unitAt<Convoy, Range>() == dimensionless);
  static_assert(unitOf<Position>() == metre && unitOf<Linear<3>>() == dimensionless);
  static_assert(unitOf<Repeated<Position>>() == metre && unitOf<Composite<Position, Landmarks>>() == metre);
  static_assert(unitOf<FromScratch>() == dimensionless &&
                unitAt<Composite<FromScratch, Position>, FromScratch>() == dimensionless);
  static_assert(!internal::HasUniformUnit<Fleet> && internal::HasUniformUnit<Composite<Position, Landmarks>>);
}

// Dual keeps the structure and inverts the units: a tag path into the space names the same block of its dual, with the
// inverse unit; the dual of a dual is the space; and a space whose every unit is dimensionless, tagged or not, is its
// own dual, so a plain matrix's spaces need no special case.
void dual_inverts_units_and_keeps_the_structure() {
  static_assert(std::is_same_v<Dual<Linear<3>>, Linear<3>>);
  static_assert(std::is_same_v<Dual<Range>, Range>);
  static_assert(std::is_same_v<Dual<Composite<Range, RadiansPi<>>>, Composite<Range, RadiansPi<>>>);
  static_assert(std::is_same_v<Dual<Position>, DualSpace<Position>>);
  static_assert(std::is_same_v<Dual<Dual<Position>>, Position>);
  static_assert(std::is_same_v<Dual<State>, DualSpace<State>> && std::is_same_v<Dual<Dual<State>>, State>);
  static_assert(Space<Dual<State>> && isLinear<Dual<State>> && Space<Dual<Map>>);
  static_assert(std::is_empty_v<Dual<State>>);
  static_assert(valueDimensionAtCompileTime<Dual<State>> == 7 && valueDimensionAtCompileTime<Dual<Map>> == Dynamic);

  static_assert(unitAt<Dual<State>, Position>() == one / metre);
  static_assert(unitAt<Dual<State>, Velocity>() == second / metre);
  static_assert(unitOf<Dual<Position>>() == one / metre);
  static_assert(unitAt<Dual<State>, Position>() * unitAt<State, Position>() == dimensionless);

  auto dual = Dual<Map>{};
  dual.resize(9);
  VERIFY_IS_EQUAL(valueDimension(dual), 9);
  VERIFY_IS_EQUAL(dual.space().block<1>().valueDimension(), 6);
}

#ifdef EIGEN_TEST_MPUNITS

namespace si = mp_units::si;

struct SiPosition : QLinear<si::metre, 3> {};
struct SiVelocity : QLinear<si::metre / si::second, 3> {};
struct SiHeading : QRadiansPi<si::radian> {};
using SiState = Composite<SiPosition, SiVelocity, SiHeading>;

// mp-units' unit objects satisfy the concept as they are, absorb Eigen's dimensionless unit, equal it exactly when
// they are mp-units' one, and type leaves, paths, and duals like the test family above.
void mp_units_units_are_units() {
  static_assert(Unit<decltype(si::metre)> && Unit<decltype(si::metre / si::second)> && Unit<decltype(mp_units::one)>);
  static_assert(si::metre * dimensionless == si::metre && dimensionless * si::metre == si::metre);
  static_assert(mp_units::one == dimensionless && !(si::metre == dimensionless));
  static_assert(internal::unitsEqual(mp_units::one, dimensionless));
  static_assert(internal::inverseUnit(si::metre) == mp_units::one / si::metre);
  static_assert(si::metre * internal::inverseUnit(si::metre) == dimensionless);

  static_assert(Space<SiState> && std::is_empty_v<SiState>);
  static_assert(unitAt<SiState, SiPosition>() == si::metre);
  static_assert(unitAt<SiState, SiVelocity>() == si::metre / si::second);
  static_assert(unitAt<SiState, SiHeading>() == si::radian);
  static_assert(std::is_same_v<Dual<Dual<SiState>>, SiState>);
  static_assert(unitAt<Dual<SiState>, SiVelocity>() == si::second / si::metre);
  static_assert(unitOf<Dual<SiPosition>>() * unitOf<SiPosition>() == dimensionless);
  static_assert(DifferenceSpace<SiHeading>::unit() == si::radian);
}

// The space rules hold over mp-units' units: they compare units, so a unit spelled another way is the same unit, and
// a covariance prediction over SI spaces has the values of the plain one.
void mp_units_meet_the_space_rules() {
  static_assert(internal::spaces_equal<SiPosition, QLinear<si::metre / si::second * si::second, 3>>::value);
  static_assert(!internal::spaces_equal<SiPosition, SiVelocity>::value);
  static_assert(internal::spaces_dual<Dual<SiState>, SiState>::value &&
                !internal::spaces_dual<SiState, SiState>::value);

  using SiTransition = QMatrix<SiState, Dual<SiState>>;
  using SiCovariance = QMatrix<SiState, SiState>;
  SiTransition f = SiTransition::Identity();
  f.topRightCorner<3, 3>().diagonal().setConstant(0.1);
  SiCovariance p = SiCovariance::Identity();
  const SiCovariance q = f * p * f.transpose() + p;
  const Matrix<double, 7, 7> fPlain = Eigen::Map<const Matrix<double, 7, 7>>(f.data());
  VERIFY_IS_APPROX((Eigen::Map<const Matrix<double, 7, 7>>(q.data())),
                   (fPlain * fPlain.transpose() + Matrix<double, 7, 7>::Identity()));
}

#endif  // EIGEN_TEST_MPUNITS

}  // namespace

EIGEN_DECLARE_TEST(spaces_units) {
  CALL_SUBTEST_1(concept_accepts_units_and_rejects_numbers());
  CALL_SUBTEST_1(dimensionless_is_the_identity());
  CALL_SUBTEST_1(plain_names_are_the_dimensionless_aliases());
  CALL_SUBTEST_1(unit_leaves_are_spaces());
  CALL_SUBTEST_1(unit_at_a_path());
  CALL_SUBTEST_1(dual_inverts_units_and_keeps_the_structure());
#ifdef EIGEN_TEST_MPUNITS
  CALL_SUBTEST_2(mp_units_units_are_units());
  CALL_SUBTEST_2(mp_units_meet_the_space_rules());
#endif
}
