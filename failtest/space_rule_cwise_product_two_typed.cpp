// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A unit of exponents, enough to tell metres from metres per second.
struct TestUnit {
  int length;
  int time;
  friend constexpr TestUnit operator*(TestUnit a, TestUnit b) { return {a.length + b.length, a.time + b.time}; }
  friend constexpr TestUnit operator/(TestUnit a, TestUnit b) { return {a.length - b.length, a.time - b.time}; }
  friend constexpr bool operator==(TestUnit, TestUnit) = default;
};
constexpr TestUnit metre{1, 0};
constexpr TestUnit metrePerSecond{1, -1};

struct Position : QLinear<metre, 3> {};
struct Velocity : QLinear<metrePerSecond, 3> {};
using State = Composite<Position, Velocity>;

// A coefficient-wise product is typed only against a dimensionless operand, such as a scalar: two typed operands
// have no spaces to put the product of their units in.
int main() {
  QVector<Position> p = QVector<Position>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  p.cwiseProduct(p).eval();
#else
  (2.0 * p).eval();
  p.cwiseProduct(Vector3d::Ones()).eval();
#endif
}
