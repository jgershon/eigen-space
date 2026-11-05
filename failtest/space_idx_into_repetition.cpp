// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

struct Position : Linear<2> {};
struct Heading : RadiansPi<> {};
struct Vehicle : Composite<Position, Heading> {};
struct Fleet : Repeated<Vehicle> {};
using State = Composite<Fleet>;

// A path through a repeated block must end at a single coordinate (TagPath.h).
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
const auto index = idx<State, TagPath<Fleet, Position>>();
#else
const auto index = idx<State, TagPath<Fleet, Heading>>();
#endif

int main() {}
