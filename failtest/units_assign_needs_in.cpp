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

struct Length : QLinear<isq::length, 2> {};
struct Height : QLinear<isq::height, 2> {};

// A length is not a height unless converted explicitly, as in mp-units; a height is a length (design/decisions.md
// [heterogeneous-results]).
int main() {
  QVector<In<Length, si::metre>> length = QVector<In<Length, si::metre>>::Zero();
  QVector<In<Height, si::metre>> height = QVector<In<Height, si::metre>>::Zero();
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
  height = length;
#else
  length = height;
  height = length.in<In<Height, si::metre>>();
#endif
  return int(height(0) + length(0));
}
