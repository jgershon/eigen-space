// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct Landmarks : Linear<Dynamic, 8> {};
struct Heading : RadiansPi<> {};

// A compile-time index needs every offset fixed by the type; after a resizable block use idx(space) (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
using State = Composite<Landmarks, Heading>;
constexpr auto index = idx<State, Heading>();
#else
using State = Composite<Heading, Landmarks>;
constexpr auto index = idx<State, Heading>();
#endif

int main() {}
