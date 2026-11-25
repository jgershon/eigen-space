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
struct Velocity : QLinear<isq::speed, 3> {};
using State = Composite<Position, Velocity>;

// In binds a unit to every leaf without one; a speed has no unit among metres (design/decisions.md
// [leaf-quantities]).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
using StateSI = In<State, si::metre>;
#else
using StateSI = In<State, si::metre, si::metre / si::second>;
#endif

int main() { return int(valueDimensionAtCompileTime<StateSI>) - 6; }
