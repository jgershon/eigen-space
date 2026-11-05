// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Geometry"

using namespace Eigen;

// map() of a temporary matrix would dangle (UnitQuaternion.h).
int main() {
  Vector4d stored = Vector4d::UnitW();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  auto q = UnitQuaternion::map(Vector4d::UnitW().eval());
#else
  auto q = UnitQuaternion::map(stored);
#endif
  (void)q;
}
