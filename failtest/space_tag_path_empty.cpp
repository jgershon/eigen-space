// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct Heading : RadiansPi<> {};

// A tag path needs at least one tag (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
TagPath<> path;
#else
TagPath<Heading> path;
#endif

int main() {}
