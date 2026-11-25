// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;

// A space with units needs Eigen/Units, which defines what Core only declares (design/decisions.md
// [mp-units-dependency]); without it the space does not compile rather than meaning something else.
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  QVector<QLinear<si::metre, 3>> x = QVector<QLinear<si::metre, 3>>::Zero();
#else
  Vector3d x = Vector3d::Zero();
#endif
  return int(x(0));
}
