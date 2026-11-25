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

// An assignment needs the destination in the spaces of the source: a transition is not a covariance.
int main() {
  QMatrix<State, State> covariance = QMatrix<State, State>::Identity();
  QMatrix<State, Dual<State>> transition = QMatrix<State, Dual<State>>::Identity();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  covariance = transition;
#else
  covariance = transition * covariance * transition.transpose();
#endif
}
