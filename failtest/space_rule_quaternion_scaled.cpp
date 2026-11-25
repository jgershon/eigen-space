// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"
#include "../Eigen/Geometry"

using namespace Eigen;

struct Attitude : UnitQuaternion {};

// Scaling a quaternion's coordinates leaves the space; a vector of plain coordinates scales (design/decisions.md
// [additive-groups]).
int main() {
  QVector<Attitude> x = QVector<Attitude>::Zero();
  x(3) = 1.0;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  QVector<Attitude> y = 2.0 * x;
#else
  QVector<Attitude> y = x;
  Vector4d plain = 2.0 * Vector4d::Ones();
  y(0) = plain(0);
#endif
  return int(y(0));
}
