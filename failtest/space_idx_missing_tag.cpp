// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct Position : Linear<2> {};
struct Heading : RadiansPi<> {};
struct Range : Linear<1> {};
using State = Composite<Position, Heading>;

// The space does not contain this tag (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
constexpr auto index = idx<State, Range>();
#else
constexpr auto index = idx<State, Heading>();
#endif

int main() {}
