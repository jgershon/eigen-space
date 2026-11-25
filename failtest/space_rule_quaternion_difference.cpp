// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"
#include "../Eigen/Geometry"

using namespace Eigen;

struct Attitude : UnitQuaternion {};

// A difference of two quaternions by coordinates is not a rotation; minus() is the space's own (design/decisions.md
// [additive-groups]).
int main() {
  QVector<Attitude> x = QVector<Attitude>::Zero();
  QVector<Attitude> y = QVector<Attitude>::Zero();
  x(3) = y(3) = 1.0;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  Vector4d d = (x - y).eval();
#else
  Vector3d d = minus<Attitude>(x, y);
#endif
  return int(d(0));
}
