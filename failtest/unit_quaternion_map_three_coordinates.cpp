// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Geometry"

using namespace Eigen;

// map() takes one column of four coordinates (UnitQuaternion.h).
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Vector3d coefficients = Vector3d::Zero();
#else
  Vector4d coefficients = Vector4d::UnitW();
#endif
  auto q = UnitQuaternion::map(coefficients);
  (void)q;
}
