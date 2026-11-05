// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

using namespace Eigen;

// A space with its own plus() has a minus() too, since a residual taken by subtraction would not invert its update
// (Space.h).
struct Scale : Linear<1> {
  Linear<1> difference() const { return {}; }
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const {
    writable(out) = (x.array() * d.array().exp()).matrix();
  }
#ifndef EIGEN_SHOULD_FAIL_TO_BUILD
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const {
    writable(out) = (a.array() / b.array()).log().matrix();
  }
#endif
};

static_assert(Space<Scale>);

int main() {}
