// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Geometry"
#include "../Eigen/Units"

#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;

// A unit quaternion's closed forms read its rotation vectors in radians, so its differences must be in a unit of angle
// of magnitude one (Geometry/UnitQuaternion.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
struct Attitude : QUnitQuaternion<si::degree> {};
#else
struct Attitude : QUnitQuaternion<si::radian> {};
#endif

int main() {
  QVector<DifferenceSpace<Attitude>> step = QVector<DifferenceSpace<Attitude>>::Zero();
  return int(step(0));
}
