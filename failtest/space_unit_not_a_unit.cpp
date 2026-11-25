// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// The first argument of QLinear is what it measures, never a number (Unit.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
QLinear<3> space;
#else
QLinear<dimensionless, 3> space;
#endif

int main() {}
