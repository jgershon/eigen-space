// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A fixed size is its own maximum (Coordinates.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
Linear<2, 3> space;
#else
Linear<2, 2> space;
#endif

int main() {}
