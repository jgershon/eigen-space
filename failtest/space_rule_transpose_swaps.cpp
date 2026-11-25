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

// Transpose swaps the spaces: the transpose of a state vector pairs with a covector in Dual<State>, and a
// state vector against itself is not a product.
int main() {
  QVector<State> x = QVector<State>::Zero();
  QVector<Dual<State>> g = QVector<Dual<State>>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  double d = (x.transpose() * x).value();
#else
  double d = (g.transpose() * x).value();
#endif
  (void)d;
}
