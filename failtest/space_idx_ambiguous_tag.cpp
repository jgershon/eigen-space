// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct X : Linear<1> {};
struct Y : Linear<1> {};
struct Left : Composite<X, Y> {};
struct Right : Composite<X> {};
using Sides = Composite<Left, Right>;

// A tag that appears more than once must be qualified by an enclosing tag (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
constexpr auto index = idx<Sides, X>();
#else
constexpr auto index = idx<Sides, TagPath<Right, X> >();
#endif

int main() {}
