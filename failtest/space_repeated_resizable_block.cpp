// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A repeated block must have a fixed, nonzero size (Composite.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
Repeated<Linear<Dynamic, 8>> space;
#else
Repeated<Linear<2>> space;
#endif

int main() {}
