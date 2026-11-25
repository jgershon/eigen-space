// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Units"

#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;
namespace isq = mp_units::isq;

// Tags without units, bound to SI units by In (design/decisions.md [leaf-quantities]).
struct PositionTag : QLinear<isq::length, 3> {};
struct VelocityTag : QLinear<isq::speed, 3> {};
using Position = In<PositionTag, si::metre>;
using Velocity = In<VelocityTag, si::metre / si::second>;
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
