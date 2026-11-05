// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct Heading : RadiansPi<> {};
using State = Composite<Linear<2>, Heading>;

// A path names blocks by tag, never by a library space type (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
constexpr auto index = idx<State, Linear<2>>();
#else
constexpr auto index = idx<State, Heading>();
#endif

int main() {}
