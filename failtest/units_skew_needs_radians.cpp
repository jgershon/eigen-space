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

struct Velocity : QLinear<isq::velocity, 3> {};
using VelocitySI = In<Velocity, si::metre / si::second>;

// A rotation's generator on velocities, skew<V, V>(phi), takes a rotation vector in radians; a vector in metres
// crossed with a velocity is not a velocity (Geometry/Skew.h).
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  const QVector<QLinear<si::metre, 3>> phi(0.1, 0.2, 0.3);
#else
  const QVector<QLinear<si::radian, 3>> phi(0.1, 0.2, 0.3);
#endif
  const auto generator = skew<VelocitySI, VelocitySI>(phi);
  return int(generator(0, 1));
}
