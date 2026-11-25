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
struct Altitude : QLinear<isq::length, 1> {};
using Fix = Composite<Position, Altitude>;

// Two units of one dimension bind by tag, not by dimension (design/decisions.md [leaf-quantities]).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
using FixSI = In<Fix, si::metre, si::kilo<si::metre>>;
#else
using FixSI = In<Fix, bind<Position>(si::metre), bind<Altitude>(si::kilo<si::metre>)>;
#endif

int main() { return int(valueDimensionAtCompileTime<FixSI>) - 4; }
