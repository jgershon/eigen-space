// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A composite needs at least one block (Composite.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
Composite<> space;
#else
Composite<Linear<1> > space;
#endif

int main() {}
