// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// What the leaves of a space measure, on mp-units (design/decisions.md [mp-units-dependency], [leaf-quantities],
// [dual-spaces]): leaves that take a quantity spec, a unit, or a reference, the plain names as the dimensionless
// aliases, tags without units and In binding them by dimension or by tag, quantityAt() and unitAt() through
// composites, repeated blocks, bound spaces, and duals, and Dual's collapse rules.

// Before main.h, whose forbidden-identifier macros (I, min, max) would break mp-units' headers.
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

#include "main.h"

#include <Eigen/Units>

#include <numbers>
#include <type_traits>

namespace {

namespace si = mp_units::si;
namespace isq = mp_units::isq;

constexpr auto metrePerSecond = si::metre / si::second;

// Tags name spaces without units; a periodic leaf reads its interval in its unit, so it carries one.
struct Position : QLinear<isq::length, 3> {};
struct Velocity : QLinear<isq::speed, 3> {};
struct Heading : QRadiansPi<si::radian> {};
struct Landmarks : QLinear<isq::length, Dynamic, 64> {};
struct Range : Linear<1> {};
using State = Composite<Position, Velocity, Heading>;
using Map = Composite<Position, Landmarks>;

using StateSI = In<State, si::metre, metrePerSecond>;
using MapSI = In<Map, si::metre>;

struct Vehicle : Composite<Position, Heading> {};
struct Fleet : Repeated<Vehicle, 4> {
  using Repeated::Repeated;
};

// A space written from scratch declares no quantity, so it is dimensionless.
struct FromScratch {
  static constexpr int FixedValueDimension = 2;
  static constexpr int ValueDimensionStep = 0;
  static constexpr int MaxValueDimension = 2;
  static constexpr Index valueDimension() { return 2; }
  friend bool operator==(const FromScratch&, const FromScratch&) = default;
};

// A leaf's first argument is what it measures: a spec, which a tag leaves without a unit, a unit alone, or a
// reference. Numbers are rejected (failtest/space_unit_not_a_unit.cpp).
void leaves_measure_specs_units_and_references() {
  static_assert(Position::quantity() == isq::length);
  static_assert(QLinear<si::metre, 3>::quantity() == si::metre);
  static_assert(QLinear<isq::height[si::metre]>::quantity() == isq::height[si::metre]);
  static_assert(!internal::isBound(isq::length) && internal::isBound(si::metre) &&
                internal::isBound(isq::height[si::metre]) && internal::isBound(dimensionless));
  static_assert(!internal::isBoundSpace<State>() && internal::isBoundSpace<StateSI>());
  static_assert(internal::isBoundSpace<Linear<3>>() && internal::isBoundSpace<Heading>());
}

// Dimensionless is a plain number: of the dimensionless kind, in a unit of magnitude one. An angle is of dimension
// one but its own kind, as in mp-units, which does not add one to a number.
void dimensionless_is_a_plain_number() {
  static_assert(internal::isDimensionless(dimensionless) && internal::isDimensionless(mp_units::one));
  static_assert(!internal::isDimensionless(si::metre) && !internal::isDimensionless(si::radian));
  static_assert(!internal::isDimensionless(mp_units::percent));
  static_assert(internal::sameQuantity(mp_units::one, dimensionless) && internal::sameQuantity(si::metre, si::metre));
  static_assert(!internal::sameQuantity(si::metre, si::second) && !internal::sameQuantity(si::metre, isq::length));
  static_assert(internal::inverseQuantity(si::metre) == mp_units::one / si::metre);
  static_assert(internal::inverseQuantity(dimensionless) == dimensionless);
}

// The plain names are the dimensionless leaves, so every spelling written before units keeps its type.
void plain_names_are_the_dimensionless_aliases() {
  static_assert(std::is_same_v<Linear<3>, QLinear<dimensionless, 3>>);
  static_assert(std::is_same_v<Linear<Dynamic, 8>, QLinear<dimensionless, Dynamic, 8>>);
  static_assert(std::is_same_v<RadiansPi<2>, QRadiansPi<dimensionless, 2>>);
  static_assert(std::is_same_v<Degrees360<>, QDegrees360<dimensionless>>);
  static_assert(std::is_same_v<Periodic<{.lower = 0.0, .upper = 256.0}>,
                               QPeriodic<dimensionless, {.lower = 0.0, .upper = 256.0}>>);
  static_assert(Linear<3>::quantity() == dimensionless && RadiansPi<>::quantity() == dimensionless);
  static_assert(std::is_empty_v<Position> && std::is_empty_v<State> && std::is_empty_v<StateSI>);
}

// In binds a unit to each leaf without one: by dimension, wherever the blocks are, or by tag. Both forms give the same
// space, which keeps the tags, structure, sizes, and operations of the unbound one, and whose difference space keeps
// the units.
void in_binds_units_to_tags() {
  using ByTag = In<State, bind<Position>(si::metre), bind<Velocity>(metrePerSecond)>;
  using Reordered = In<State, metrePerSecond, si::metre>;
  static_assert(std::is_same_v<StateSI, ByTag> && std::is_same_v<StateSI, Reordered>);
  static_assert(std::is_base_of_v<State, StateSI>);
  static_assert(unitAt<StateSI, Position>() == si::metre && unitAt<StateSI, Velocity>() == metrePerSecond);
  static_assert(quantityAt<StateSI, Position>() == isq::length[si::metre]);
  static_assert(unitAt<StateSI, Heading>() == si::radian);

  // Two units of one dimension go by tag.
  using Mixed = In<Composite<Position, Landmarks>, bind<Position>(si::metre), bind<Landmarks>(si::kilo<si::metre>)>;
  static_assert(unitAt<Mixed, Landmarks>() == si::kilo<si::metre> && unitAt<Mixed, Position>() == si::metre);

  static_assert(Space<StateSI> && Space<MapSI> && isLinear<In<Position, si::metre>> && !isLinear<StateSI>);
  static_assert(valueDimensionAtCompileTime<StateSI> == 7 && valueDimensionAtCompileTime<MapSI> == Dynamic);
  using Differences = DifferenceSpace<StateSI>;
  static_assert(internal::IsBoundSpace<Differences> && !std::is_same_v<Differences, StateSI>);
  static_assert(internal::spaces_equal<Differences, StateSI>::value);

  auto x = Vector<double, 7>{1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 4.0};
  normalizeValues(StateSI{}, x);
  VERIFY(std::abs(x(6) - (4.0 - 2 * std::numbers::pi)) <= 1e-15);
  auto map = MapSI{};
  map.resize(9);
  VERIFY_IS_EQUAL(valueDimension(map), 9);
  VERIFY_IS_EQUAL(map.block<1>().valueDimension(), 6);
}

// quantityAt() and unitAt() read a block by tag path, through composites and repeated blocks; quantityOf() and unitOf()
// read a whole space where its leaves agree; a space written from scratch is dimensionless.
void quantity_at_a_path() {
  using Convoy = Composite<In<Fleet, si::metre>, Range>;
  static_assert(unitAt<Convoy, TagPath<Fleet, Heading>>() == si::radian);
  static_assert(unitAt<Convoy, Range>() == dimensionless);
  static_assert(unitOf<In<Position, si::metre>>() == si::metre && unitOf<Linear<3>>() == dimensionless);
  static_assert(unitOf<In<Repeated<Position>, si::metre>>() == si::metre && unitOf<MapSI>() == si::metre);
  static_assert(unitOf<FromScratch>() == dimensionless &&
                unitAt<Composite<FromScratch, Heading>, FromScratch>() == dimensionless);
  static_assert(!internal::HasUniformQuantity<In<Fleet, si::metre>> && internal::HasUniformQuantity<MapSI>);
}

// Dual keeps the structure and inverts the units: a tag path into the space names the same block of its dual; the
// dual of a dual is the space; and a dimensionless space, tagged or not, is its own dual.
void dual_inverts_units_and_keeps_the_structure() {
  static_assert(std::is_same_v<Dual<Linear<3>>, Linear<3>>);
  static_assert(std::is_same_v<Dual<Range>, Range>);
  static_assert(std::is_same_v<Dual<Composite<Range, RadiansPi<>>>, Composite<Range, RadiansPi<>>>);
  static_assert(std::is_same_v<Dual<StateSI>, DualSpace<StateSI>> && std::is_same_v<Dual<Dual<StateSI>>, StateSI>);
  static_assert(std::is_same_v<Dual<Heading>, DualSpace<Heading>>);
  static_assert(Space<Dual<StateSI>> && isLinear<Dual<StateSI>> && std::is_empty_v<Dual<StateSI>>);
  static_assert(valueDimensionAtCompileTime<Dual<StateSI>> == 7 && valueDimensionAtCompileTime<Dual<MapSI>> == Dynamic);

  static_assert(unitAt<Dual<StateSI>, Position>() == mp_units::one / si::metre);
  static_assert(unitAt<Dual<StateSI>, Velocity>() == si::second / si::metre);
  static_assert(unitAt<Dual<StateSI>, Position>() * unitAt<StateSI, Position>() == mp_units::one);

  auto dual = Dual<MapSI>{};
  dual.resize(9);
  VERIFY_IS_EQUAL(valueDimension(dual), 9);
  VERIFY_IS_EQUAL(dual.space().block<1>().valueDimension(), 6);
}

}  // namespace

EIGEN_DECLARE_TEST(spaces_units) {
  CALL_SUBTEST_1(leaves_measure_specs_units_and_references());
  CALL_SUBTEST_1(dimensionless_is_a_plain_number());
  CALL_SUBTEST_1(plain_names_are_the_dimensionless_aliases());
  CALL_SUBTEST_1(in_binds_units_to_tags());
  CALL_SUBTEST_1(quantity_at_a_path());
  CALL_SUBTEST_1(dual_inverts_units_and_keeps_the_structure());
}
