// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A unit of exponents, enough to give two blocks different units.
struct TestUnit {
  int length;
  friend constexpr TestUnit operator*(TestUnit a, TestUnit b) { return {a.length + b.length}; }
  friend constexpr TestUnit operator/(TestUnit a, TestUnit b) { return {a.length - b.length}; }
  friend constexpr bool operator==(TestUnit, TestUnit) = default;
};
constexpr TestUnit metre{1};

struct Position : QLinear<metre, 3> {};
struct Count : Linear<2> {};
struct Mixed : Composite<Position, Count> {};
using State = Composite<Mixed>;

// A block whose coordinates have different units has no unit of its own (Dual.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
constexpr auto unit = unitAt<State, Mixed>();
#else
constexpr auto unit = unitAt<State, TagPath<Mixed, Position> >();
#endif

int main() {}
