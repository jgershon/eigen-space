// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// checkGroup(): a LieGroup's members checked against each other on plain coordinates, as EmrysFilter's
// groups-prototype checks them (Spaces/Group.h). The right Jacobian is checked against the derivative of
// Log(Exp(d)^-1 Exp(d + e)) in e at zero, by automatic differentiation.

#ifndef EIGEN_TEST_LIE_GROUP_HELPERS_H
#define EIGEN_TEST_LIE_GROUP_HELPERS_H

#include <Eigen/AutoDiff>

#include <algorithm>

namespace lie_group_test {

/** The largest residual of each group law, in the coordinates of a difference. */
struct GroupCheck {
  double identity = 0;       // a e and e a against a
  double inverse = 0;        // a a^-1 against e
  double associative = 0;    // (a b) c against a (b c), with c = Exp(d)
  double logOfExp = 0;       // Log(Exp(d)) against d
  double plus = 0;           // plus(a, d) against a Exp(d)
  double minus = 0;          // minus(a, b) against Log(b^-1 a)
  double adjoint = 0;        // a Exp(d) against Exp(Ad_a d) a
  double rightJacobian = 0;  // J_r(d) against the derivative of Log(Exp(d)^-1 Exp(d + e))

  double worst() const {
    return (std::max)({identity, inverse, associative, logOfExp, plus, minus, adjoint, rightJacobian});
  }
};

template <typename S, typename A, typename B, typename D>
GroupCheck checkGroup(const S& space, const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b,
                      const Eigen::MatrixBase<D>& d) {
  using Eigen::Index;
  using Value = Eigen::VectorXd;
  const auto distance = [&](const Value& x, const Value& y) {
    const Value error = Eigen::minus(space, x, y);
    return error.size() == 0 ? 0.0 : double(error.cwiseAbs().maxCoeff());
  };
  const auto size = [](const Value& v) { return v.size() == 0 ? 0.0 : double(v.cwiseAbs().maxCoeff()); };
  const Value va = a;
  const Value vb = b;
  const Value step = d;
  const Value e = Eigen::identity(space);
  const Value c = Eigen::exp(space, step);

  GroupCheck result;
  result.identity = (std::max)(distance(Eigen::compose(space, va, e), va), distance(Eigen::compose(space, e, va), va));
  result.inverse = distance(Eigen::compose(space, va, Eigen::inverse(space, va)), e);
  result.associative = distance(Eigen::compose(space, Value{Eigen::compose(space, va, vb)}, c),
                                Eigen::compose(space, va, Value{Eigen::compose(space, vb, c)}));
  result.logOfExp = size(Value{Eigen::log(space, c) - step});
  result.plus = distance(Value{Eigen::plus(space, va, step)}, Eigen::compose(space, va, c));
  result.minus = size(Value{Eigen::minus(space, va, vb) -
                            Eigen::log(space, Value{Eigen::compose(space, Value{Eigen::inverse(space, vb)}, va)})});
  const Value turned = Eigen::adjoint(space, va) * step;
  result.adjoint = distance(Eigen::compose(space, va, c), Eigen::compose(space, Value{Eigen::exp(space, turned)}, va));

  using Dual = Eigen::AutoDiffScalar<Eigen::VectorXd>;
  using DualVector = Eigen::Matrix<Dual, Eigen::Dynamic, 1>;
  const Index n = differenceDimension(space);
  DualVector perturbed(n);
  for (Index i = 0; i < n; ++i) {
    perturbed(i) = Dual{step(i), Eigen::VectorXd::Unit(n, i)};
  }
  const DualVector moved = Eigen::exp(space, perturbed);
  const DualVector back = Value{Eigen::inverse(space, c)}.template cast<Dual>();
  const DualVector error = Eigen::log(space, DualVector{Eigen::compose(space, back, moved)});
  const Eigen::MatrixXd jacobian = Eigen::rightJacobian(space, step);
  for (Index r = 0; r < n; ++r) {
    for (Index k = 0; k < n; ++k) {
      const double derivative = error(r).derivatives().size() == n ? error(r).derivatives()(k) : 0.0;
      result.rightJacobian = (std::max)(result.rightJacobian, std::abs(jacobian(r, k) - derivative));
    }
  }
  return result;
}

}  // namespace lie_group_test

#endif  // EIGEN_TEST_LIE_GROUP_HELPERS_H
