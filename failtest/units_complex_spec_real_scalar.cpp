// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Units"

#include <complex>
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

using namespace Eigen;
namespace si = mp_units::si;
namespace isq = mp_units::isq;

struct Phasors : QLinear<isq::voltage_phasor, 2> {};

// A phasor is complex, so its coordinates hold a complex scalar, as mp-units' RepresentationOf checks
// (design/decisions.md [complex-scalars]).
int main() {
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  QVector<In<Phasors, si::volt>, double> v = QVector<In<Phasors, si::volt>, double>::Zero();
  return int(v(0));
#else
  QVector<In<Phasors, si::volt>, std::complex<double> > v =
      QVector<In<Phasors, si::volt>, std::complex<double> >::Zero();
  return int(v(0).real());
#endif
}
