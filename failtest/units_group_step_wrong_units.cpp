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
using BodySI = In<Body, si::metre / si::second, si::metre>;

// exp() of the group takes a step of its difference space, whose velocity rows are in metres per second: a step whose
// velocity rows are in metres does not convert (Spaces/LieGroup.h).
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  using Step = QVector<Composite<QLinear<si::radian, 3>, QLinear<si::metre, 3>, QLinear<si::metre, 3>>>;
#else
  using Step = QVector<DifferenceSpace<BodySI>>;
#endif
  const Step step = Step::Zero();
  const QVector<BodySI> x = exp<BodySI>(step);
  return int(x(0));
}
