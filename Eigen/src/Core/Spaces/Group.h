// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_GROUP_H
#define EIGEN_SPACES_GROUP_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal Whether \a S has a group's members of its own: compose(a, b, out), inverse(a, out), identity(out),
 * exp(d, out), log(a, out), adjoint(a, out), and rightJacobian(d, out), each const, taking one column per operand and
 * writing through \a out as plus() does.
 */
template <typename S>
concept HasGroupMembers = requires(const S& space, MatrixXd& m) {
  space.compose(m.col(0), m.col(0), m.col(0));
  space.inverse(m.col(0), m.col(0));
  space.identity(m.col(0));
  space.exp(m.col(0), m.col(0));
  space.log(m.col(0), m.col(0));
  space.adjoint(m.col(0), m.middleRows(0, 0));
  space.rightJacobian(m.col(0), m.middleRows(0, 0));
};

template <typename S>
consteval bool isAdditiveGroup();

/** \internal isAdditiveGroup() over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct AllAdditive;

template <typename... Blocks>
struct AllAdditive<std::tuple<Blocks...>> : std::bool_constant<(isAdditiveGroup<Blocks>() && ...)> {};

/** \internal Whether \a S is a group under addition, so that +, -, and the coefficient-wise operations on its
 * coordinates are its own operations (design/decisions.md [additive-groups]): a leaf whose values and differences
 * are plain, or a Periodic one, which adds exactly modulo its period; a Composite or Repeated of such; a dual, which
 * is linear [dual-spaces]; and a bound space if its unbound one is. The classification is that of EmrysFilter's
 * groups-prototype, where Linear and Periodic are the groups under addition; a UnitQuaternion, a space with a plus()
 * of its own, or a group with its own product, such as an ExtendedPose of a heading and a vector, is not one.
 */
template <typename S>
consteval bool isAdditiveGroup() {
  if constexpr (IsBoundSpace<S>) {
    return isAdditiveGroup<BoundBase<S>>();
  } else if constexpr (IsDualSpace<S>) {
    return true;
  } else if constexpr (HasGroupMembers<S>) {
    return false;
  } else if constexpr (IsComposite<S>) {
    return AllAdditive<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return isAdditiveGroup<RepeatedBlock<S>>();
  } else {
    return isLinear<S> || IsPeriodic<S>;
  }
}

template <typename S>
consteval bool isLieGroup();

/** \internal isLieGroup() over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct AllLieGroups;

template <typename... Blocks>
struct AllLieGroups<std::tuple<Blocks...>> : std::bool_constant<(isLieGroup<Blocks>() && ...)> {};

/** \internal Whether \a S is a Lie group: it has the members; or it is a Composite or Repeated of groups; or it is a
 * group under addition. A space with a plus() of its own and no group members is not one.
 */
template <typename S>
consteval bool isLieGroup() {
  if constexpr (IsBoundSpace<S>) {
    return isLieGroup<BoundBase<S>>();
  } else if constexpr (HasGroupMembers<S>) {
    return true;
  } else if constexpr (IsComposite<S>) {
    return AllLieGroups<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return isLieGroup<RepeatedBlock<S>>();
  } else {
    return isLinear<S> || IsPeriodic<S>;
  }
}

template <typename S>
consteval bool isAbelianGroup();

/** \internal isAbelianGroup() over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
struct AllAbelian;

template <typename... Blocks>
struct AllAbelian<std::tuple<Blocks...>> : std::bool_constant<(isAbelianGroup<Blocks>() && ...)> {};

/** \internal Whether the group \a S is abelian, so that its adjoint is the identity: a group under addition, a
 * Composite or Repeated of such, or a group that declares <tt>static constexpr bool abelian = true;</tt>.
 */
template <typename S>
consteval bool isAbelianGroup() {
  if constexpr (IsBoundSpace<S>) {
    return isAbelianGroup<BoundBase<S>>();
  } else if constexpr (HasGroupMembers<S>) {
    return requires { requires bool(S::abelian); };
  } else if constexpr (IsComposite<S>) {
    return AllAbelian<CompositeBlocks<S>>::value;
  } else if constexpr (IsRepeated<S>) {
    return isAbelianGroup<RepeatedBlock<S>>();
  } else {
    return true;
  }
}

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief A Space that is a Lie group: two values combine into a third, every value has an inverse, and one value is
 * the identity.
 *
 * In a group the product determines the space's operations,
 * \f[ x \boxplus \delta = x \cdot \mathrm{Exp}(\delta), \qquad a \boxminus b = \mathrm{Log}(b^{-1} \cdot a), \f]
 * with the increment on the right, so that a difference is a step in the body frame, as in
 * \ref Sola2018 "Solà, Deray, and Atchuthan (2018)". Linear and Periodic are groups under addition; UnitQuaternion and
 * ExtendedPose have their own products; a Composite or Repeated of groups is one block by block. A space with a
 * plus() of its own that is not a group, such as a unit direction, stays a Space. The design is that of EmrysFilter's
 * groups-prototype (design/decisions.md [additive-groups]).
 *
 * A group written from scratch has, beside what Space asks for, these const members, each taking one column per
 * operand, in any scalar, and writing through \a out as plus() does:
 *  - <tt>compose(a, b, out)</tt>, the product \f$ a \cdot b \f$;
 *  - <tt>inverse(a, out)</tt> and <tt>identity(out)</tt>;
 *  - <tt>exp(d, out)</tt>, the value \f$ \mathrm{Exp}(d) \f$ of a difference, and <tt>log(a, out)</tt>, its inverse;
 *  - <tt>adjoint(a, out)</tt>, the matrix \f$ \mathrm{Ad}_a \f$ over the differences with
 *    \f$ a \cdot \mathrm{Exp}(d) = \mathrm{Exp}(\mathrm{Ad}_a d) \cdot a \f$;
 *  - <tt>rightJacobian(d, out)</tt>, \f$ J_r(d) \f$ with
 *    \f$ \mathrm{Exp}(d + \epsilon) \approx \mathrm{Exp}(d) \cdot \mathrm{Exp}(J_r(d) \epsilon) \f$.
 *
 * Its plus() and minus() must be the two formulas above.
 */
template <typename S>
concept LieGroup = Space<S> && internal::isLieGroup<S>();

/** \ingroup Spaces_Module
 * Whether the adjoint of the group \a S is the identity at every value: true of a group under addition and of a
 * Composite or Repeated of such.
 */
template <LieGroup S>
inline constexpr bool hasIdentityAdjoint = internal::isAbelianGroup<S>();

namespace internal {

/** \internal Calls \a f(block, valueOffset, valueRows, differenceOffset, differenceRows) for each block of the
 * Composite \a space, or each repetition of the Repeated \a space, in order.
 */
template <typename S, typename F>
void forEachGroupPart(const S& space, F&& f) {
  if constexpr (IsComposite<S>) {
    Index values = 0;
    Index differences = 0;
    [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      ((f(space.template block<Idx>(), values, valueDimension(space.template block<Idx>()), differences,
          differenceDimension(space.template block<Idx>())),
        values += valueDimension(space.template block<Idx>()),
        differences += differenceDimension(space.template block<Idx>())),
       ...);
    }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<S>>>{});
  } else {
    const auto& block = space.block();
    const Index v = valueDimension(block);
    const Index k = differenceDimension(block);
    for (Index i = 0; i < space.count(); ++i) {
      f(block, i * v, v, i * k, k);
    }
  }
}

/** \internal Whether the group \a S is one block by block: a Composite or Repeated without group members of its own.
 */
template <typename S>
concept IsBlockwiseGroup = !HasGroupMembers<S> && (IsComposite<S> || IsRepeated<S>);

/** \internal Writes the product \f$ a \cdot b \f$ in the group \a space through \a out. */
template <typename S, typename A, typename B, typename Out>
void composeInto(const S& space, const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) {
  using T = typename Out::Scalar;
  if constexpr (HasGroupMembers<S>) {
    space.compose(a, b, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    forEachGroupPart(space, [&](const auto& block, Index v, Index n, Index, Index) {
      composeInto(block, a.segment(v, n), b.segment(v, n), o.segment(v, n));
    });
  } else {
    writable(out) = a.template cast<T>() + b.template cast<T>();
    normalizeValues(space, out);
  }
}

/** \internal Writes the inverse of \a a in the group \a space through \a out. */
template <typename S, typename A, typename Out>
void inverseInto(const S& space, const MatrixBase<A>& a, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.inverse(a, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    forEachGroupPart(space, [&](const auto& block, Index v, Index n, Index, Index) {
      inverseInto(block, a.segment(v, n), o.segment(v, n));
    });
  } else {
    writable(out) = -a.template cast<typename Out::Scalar>();
    normalizeValues(space, out);
  }
}

/** \internal Writes the identity of the group \a space through \a out. */
template <typename S, typename Out>
void identityInto(const S& space, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.identity(out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    forEachGroupPart(space,
                     [&](const auto& block, Index v, Index n, Index, Index) { identityInto(block, o.segment(v, n)); });
  } else {
    writable(out).setZero();
    normalizeValues(space, out);
  }
}

/** \internal Writes \f$ \mathrm{Exp}(d) \f$ in the group \a space through \a out. */
template <typename S, typename D, typename Out>
void expInto(const S& space, const MatrixBase<D>& d, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.exp(d, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    forEachGroupPart(space, [&](const auto& block, Index v, Index n, Index dv, Index dn) {
      expInto(block, d.segment(dv, dn), o.segment(v, n));
    });
  } else {
    writable(out) = d.template cast<typename Out::Scalar>();
    normalizeValues(space, out);
  }
}

/** \internal Writes \f$ \mathrm{Log}(a) \f$ in the group \a space through \a out. */
template <typename S, typename A, typename Out>
void logInto(const S& space, const MatrixBase<A>& a, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.log(a, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    forEachGroupPart(space, [&](const auto& block, Index v, Index n, Index dv, Index dn) {
      logInto(block, a.segment(v, n), o.segment(dv, dn));
    });
  } else {
    writable(out) = a.template cast<typename Out::Scalar>();
    normalizeDifferences(space, out);
  }
}

/** \internal Writes the adjoint \f$ \mathrm{Ad}_a \f$ of the group \a space through \a out, a square matrix over its
 * differences.
 */
template <typename S, typename A, typename Out>
void adjointInto(const S& space, const MatrixBase<A>& a, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.adjoint(a, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    o.setZero();
    forEachGroupPart(space, [&](const auto& block, Index v, Index n, Index dv, Index dn) {
      adjointInto(block, a.segment(v, n), o.block(dv, dv, dn, dn));
    });
  } else {
    writable(out).setIdentity();
  }
}

/** \internal Writes the right Jacobian \f$ J_r(d) \f$ of the group \a space through \a out, a square matrix over its
 * differences.
 */
template <typename S, typename D, typename Out>
void rightJacobianInto(const S& space, const MatrixBase<D>& d, const MatrixBase<Out>& out) {
  if constexpr (HasGroupMembers<S>) {
    space.rightJacobian(d, out);
  } else if constexpr (IsBlockwiseGroup<S>) {
    auto& o = writable(out);
    o.setZero();
    forEachGroupPart(space, [&](const auto& block, Index, Index, Index dv, Index dn) {
      rightJacobianInto(block, d.segment(dv, dn), o.block(dv, dv, dn, dn));
    });
  } else {
    writable(out).setIdentity();
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_GROUP_H
