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

struct Velocity : QLinear<isq::velocity, 3> {};
struct Position : QLinear<isq::displacement, 3> {};
using State = In<Composite<Velocity, Position>, si::metre / si::second, si::metre>;

// A block by tag is typed by the subspace it names, so a write into the velocity's rows must be a velocity
// (design/decisions.md [typed-blocks]).
int main() {
  QVector<State> x = QVector<State>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  x.segment<Velocity>() = QVector<In<Position, si::metre>>(1.0, 2.0, 3.0);
#else
  x.segment<Position>() = QVector<In<Position, si::metre>>(1.0, 2.0, 3.0);
#endif
  return int(x(3));
}
