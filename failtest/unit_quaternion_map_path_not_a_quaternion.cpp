// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Geometry"

using namespace Eigen;

struct Position : Linear<2> {};
struct Heading : RadiansPi<> {};
struct Attitude : UnitQuaternion {};
using State = Composite<Position, Heading, Attitude>;

// The path must name a UnitQuaternion block (UnitQuaternion.h).
int main() {
  Matrix<double, 7, 1> x = Matrix<double, 7, 1>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  auto q = UnitQuaternion::map<Heading>(State{}, x);
#else
  auto q = UnitQuaternion::map<Attitude>(State{}, x);
#endif
  (void)q;
}
