// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// The upper end must be above the lower (Coordinates.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
Periodic<{.lower = 1.0, .upper = 0.0}> space;
#else
Periodic<{.lower = 0.0, .upper = 1.0}> space;
#endif

int main() {}
