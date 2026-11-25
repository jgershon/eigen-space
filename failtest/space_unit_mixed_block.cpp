// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Units"

#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;
namespace isq = mp_units::isq;

struct Position : QLinear<isq::length, 3> {};
struct Count : Linear<2> {};
struct Mixed : Composite<Position, Count> {};
using State = In<Composite<Mixed>, si::metre>;

// A block whose coordinates measure different quantities has no unit of its own (Dual.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
constexpr auto unit = unitAt<State, Mixed>();
#else
constexpr auto unit = unitAt<State, TagPath<Mixed, Position> >();
#endif

int main() {}
