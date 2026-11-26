// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Geometry"
#include "../Eigen/Units"

#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;
namespace isq = mp_units::isq;

struct Attitude : QUnitQuaternion<si::radian> {};
struct Velocity : QLinear<isq::velocity, 3> {};
struct Position : QLinear<isq::displacement, 3> {};
struct Body : ExtendedPose<Attitude, Velocity, Position> {};
using Step = DifferenceSpace<In<Body, si::metre / si::second, si::metre>>;
struct Sighting : QLinear<isq::displacement, 3> {};
using SightingSI = In<Sighting, si::metre>;

// The invariant EKF's measurement Jacobian of a landmark seen from the body: its attitude block is [y]x in metres per
// radian. The position's block, -I in metres per metre, written into the attitude's columns does not compile
// (test/invariant_ekf.cpp, design/decisions.md [lie-groups]).
int main() {
  const QVector<SightingSI> predicted(1.0, 2.0, 3.0);
  QMatrix<SightingSI, Dual<Step>> h = QMatrix<SightingSI, Dual<Step>>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  h.middleCols<Attitude>() = -QMatrix<SightingSI, Dual<In<Position, si::metre>>>::Identity();
#else
  h.middleCols<Attitude>() = skew<QLinear<si::radian, 3>>(predicted);
#endif
  return int(h(0, 1));
}
