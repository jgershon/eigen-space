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

struct Position : QLinear<isq::displacement, 3> {};
struct Velocity : QLinear<isq::velocity, 3> {};
using State = In<Composite<Position, Velocity>, si::metre, si::metre / si::second>;

// A matrix's axes may differ from the destination's by one common factor c (design/decisions.md [dual-spaces]):
// sigma^2 I in m^2 by 1 is a position covariance. In a covariance over positions and velocities, no one c serves both
// blocks, so a variance in m^2 throughout does not assign.
int main() {
  QMatrix<State, State> p = QMatrix<State, State>::Zero();
  const auto variance = 2.0 * si::metre * si::metre;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  p = variance * Matrix<double, 6, 6>::Identity();
#else
  p.block<Position, Position>() = variance * Matrix3d::Identity();
#endif
  return int(p(0, 0));
}
