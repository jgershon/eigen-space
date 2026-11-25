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

struct Position : QLinear<isq::length, 3> {};

// A tag names a space without units; a matrix's spaces need them (design/decisions.md [leaf-quantities]).
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  QVector<Position> x = QVector<Position>::Zero();
#else
  QVector<In<Position, si::metre> > x = QVector<In<Position, si::metre> >::Zero();
#endif
  return int(x(0));
}
