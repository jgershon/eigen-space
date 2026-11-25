// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_RULES_H
#define EIGEN_SPACES_RULES_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// The space rules of design/decisions.md [dual-spaces], after Hart, Multidimensional Analysis (Springer, 1995),
// chapters 2 to 4. Entry (i, j) of a matrix over (R, C) has the unit u_R(i) u_C(j), so
//   A + B:  u_RA(i) u_CA(j) = u_RB(i) u_CB(j) for all i, j, which is RA = RB and CA = CB up to a common factor;
//   A * B:  u_RA(i) u_CA(k) u_RB(k) u_CB(j) must not depend on k, which CA = Dual<RB> meets.
// The rules compare units, coordinate by coordinate, not the space types: a tag, a composite's nesting, and
// periodicity address and wrap coordinates but do not change what they are measured in. The common factor is
// not admitted: a sum of a matrix and a scaled copy of it in other units is rejected, as the decision records.

/** \internal \a Size coordinates (\c Dynamic if resizable) in the unit \a U. */
template <auto U, int Size>
struct UnitRun {};

/** \internal A repeated block whose coordinates are in more than one unit: the runs of one repetition, repeated a
 * number of times known only at run time. */
template <typename BlockRuns>
struct RepeatedRuns {};

/** \internal The units of a space, coordinate by coordinate, as runs: no run empty, and no two adjacent unit runs in
 * one unit. That form is unique for fixed sizes, so two spaces measure their coordinates alike when their run lists
 * match pairwise. A resizable run matches any size, as Linear<Dynamic> does Linear<n>; the sizes are then checked
 * at run time, in total only, so a run-time boundary between units is not checked [typed-blocks]. */
template <typename... Runs>
struct RunList {};

constexpr int addRunSizes(int a, int b) { return a == Dynamic || b == Dynamic ? Dynamic : a + b; }

/** \internal The run list of \a Run in front of the run list \a List, merged with its first run where the units are
 * equal. */
template <typename Run, typename List>
struct PrependRun;

template <typename Run, typename... Runs>
struct PrependRun<Run, RunList<Runs...>> {
  using type = RunList<Run, Runs...>;
};

template <auto U, int N, typename... Runs>
struct PrependRun<UnitRun<U, N>, RunList<Runs...>> {
  using type = std::conditional_t<N == 0, RunList<Runs...>, RunList<UnitRun<U, N>, Runs...>>;
};

template <auto U, int N, auto V, int M, typename... Rest>
struct PrependRun<UnitRun<U, N>, RunList<UnitRun<V, M>, Rest...>> {
  using type = std::conditional_t<N == 0, RunList<UnitRun<V, M>, Rest...>,
                                  std::conditional_t<unitsEqual(U, V), RunList<UnitRun<U, addRunSizes(N, M)>, Rest...>,
                                                     RunList<UnitRun<U, N>, UnitRun<V, M>, Rest...>>>;
};

/** \internal The run list of the coordinates of \a Front followed by those of \a Back. */
template <typename Front, typename Back>
struct ConcatRuns;

template <typename Back>
struct ConcatRuns<RunList<>, Back> {
  using type = Back;
};

template <typename First, typename... Rest, typename Back>
struct ConcatRuns<RunList<First, Rest...>, Back> {
  using type = typename PrependRun<First, typename ConcatRuns<RunList<Rest...>, Back>::type>::type;
};

template <typename List>
struct InvertRuns;

/** \internal A run in the inverse units: a coordinate of the dual space. */
template <typename Run>
struct InvertRun;

template <auto U, int N>
struct InvertRun<UnitRun<U, N>> {
  using type = UnitRun<inverseUnit(U), N>;
};

template <typename BlockRuns>
struct InvertRun<RepeatedRuns<BlockRuns>> {
  using type = RepeatedRuns<typename InvertRuns<BlockRuns>::type>;
};

/** \internal The run list of the dual: every unit inverted. Distinct units have distinct inverses, so the form is
 * kept. */
template <typename... Runs>
struct InvertRuns<RunList<Runs...>> {
  using type = RunList<typename InvertRun<Runs>::type...>;
};

template <typename S>
auto spaceRuns();

/** \internal The run list of the space \a S. */
template <typename S>
using SpaceRuns = typename decltype(spaceRuns<S>())::type;

/** \internal The runs of the blocks in the tuple type \a Blocks, in order. */
template <typename Blocks>
struct BlockRuns;

template <>
struct BlockRuns<std::tuple<>> {
  using type = RunList<>;
};

template <typename First, typename... Rest>
struct BlockRuns<std::tuple<First, Rest...>> {
  using type = typename ConcatRuns<SpaceRuns<First>, typename BlockRuns<std::tuple<Rest...>>::type>::type;
};

/** \internal \returns the run list of \a S, as a std::type_identity: a dual's inverted, a composite's blocks' in
 * order, a repeated block's as one resizable run where it has one unit, and a leaf's as one run of its size. */
template <typename S>
auto spaceRuns() {
  if constexpr (IsDualSpace<S>) {
    return std::type_identity<typename InvertRuns<SpaceRuns<DualInner<S>>>::type>{};
  } else if constexpr (IsComposite<S>) {
    return std::type_identity<typename BlockRuns<CompositeBlocks<S>>::type>{};
  } else if constexpr (IsRepeated<S>) {
    using Block = RepeatedBlock<S>;
    if constexpr (hasUniformUnit<Block>()) {
      return std::type_identity<RunList<UnitRun<firstUnit<Block>(), Dynamic>>>{};
    } else {
      return std::type_identity<RunList<RepeatedRuns<SpaceRuns<Block>>>>{};
    }
  } else {
    return std::type_identity<
        typename PrependRun<UnitRun<leafUnit<S>(), valueDimensionAtCompileTime<S>>, RunList<>>::type>{};
  }
}

template <typename A, typename B>
struct RunsMatch;

/** \internal Whether two runs measure their coordinates alike: the same unit, and sizes that can agree. */
template <typename A, typename B>
struct RunMatches : std::false_type {};

template <auto U, int N, auto V, int M>
struct RunMatches<UnitRun<U, N>, UnitRun<V, M>>
    : std::bool_constant<unitsEqual(U, V) && (N == M || N == Dynamic || M == Dynamic)> {};

template <typename A, typename B>
struct RunMatches<RepeatedRuns<A>, RepeatedRuns<B>> : RunsMatch<A, B> {};

/** \internal Whether two run lists match pairwise. */
template <typename A, typename B>
struct RunsMatch : std::false_type {};

template <typename... A, typename... B>
  requires(sizeof...(A) == sizeof...(B))
struct RunsMatch<RunList<A...>, RunList<B...>> : std::bool_constant<(RunMatches<A, B>::value && ...)> {};

/** \internal Whether every coordinate of \a S is dimensionless; immediate for a plain Linear, the space of every
 * untyped matrix, so that untyped code skips the rules. */
template <typename S>
inline constexpr bool is_dimensionless_space = isDimensionlessSpace<S>();

template <int N, int M>
inline constexpr bool is_dimensionless_space<QLinear<dimensionless, N, M>> = true;

/** \internal Whether the spaces \a A and \a B measure every coordinate alike, the rule of a sum. */
template <typename A, typename B>
struct spaces_equal : std::disjunction<std::is_same<A, B>, RunsMatch<SpaceRuns<A>, SpaceRuns<B>>> {};

/** \internal Whether \a A is the dual of \a B, coordinate by coordinate, the rule of a product's inner spaces. */
template <typename A, typename B>
struct spaces_dual : RunsMatch<SpaceRuns<A>, typename InvertRuns<SpaceRuns<B>>::type> {};

/** \internal Whether an expression of type \a T has a unit anywhere. An expression that spaces_of is not
 * specialized for has dimensionless Linear spaces, so it has none, and meets the rules only with untyped operands. */
template <typename T>
inline constexpr bool has_typed_spaces = !(is_dimensionless_space<typename spaces_of<T>::RowSpace> &&
                                           is_dimensionless_space<typename spaces_of<T>::ColSpace>);

/** \internal Whether the expressions \a A and \a B have equal row spaces and equal column spaces, immediately where
 * neither has a unit. */
template <typename A, typename B>
struct xpr_spaces_equal
    : std::disjunction<
          std::bool_constant<!has_typed_spaces<A> && !has_typed_spaces<B>>,
          std::conjunction<spaces_equal<typename spaces_of<A>::RowSpace, typename spaces_of<B>::RowSpace>,
                           spaces_equal<typename spaces_of<A>::ColSpace, typename spaces_of<B>::ColSpace>>> {};

/** \internal How a coefficient-wise operation combines units: equal operands (a sum, a difference, a comparison, a
 * minimum, and any operation not listed, which is the conservative default), or a product or quotient, whose units
 * multiply and which is typed here only where the second operand (for a product, either one) is dimensionless, as a
 * scalar is. Two typed operands of a coefficient-wise product have no spaces to put the product in. */
enum class unit_combination { equal, product, quotient };

template <typename Op>
struct binary_op_unit_combination : std::integral_constant<unit_combination, unit_combination::equal> {};
template <typename A, typename B>
struct binary_op_unit_combination<scalar_product_op<A, B>>
    : std::integral_constant<unit_combination, unit_combination::product> {};
template <typename A, typename B>
struct binary_op_unit_combination<scalar_conj_product_op<A, B>>
    : std::integral_constant<unit_combination, unit_combination::product> {};
template <typename A, typename B>
struct binary_op_unit_combination<scalar_quotient_op<A, B>>
    : std::integral_constant<unit_combination, unit_combination::quotient> {};

template <typename Func>
struct assign_op_unit_combination : std::integral_constant<unit_combination, unit_combination::equal> {};
template <typename A, typename B>
struct assign_op_unit_combination<mul_assign_op<A, B>>
    : std::integral_constant<unit_combination, unit_combination::quotient> {};
template <typename A, typename B>
struct assign_op_unit_combination<div_assign_op<A, B>>
    : std::integral_constant<unit_combination, unit_combination::quotient> {};

template <typename Lhs, typename Rhs>
struct product_spaces {
  using LhsInner = typename spaces_of<Lhs>::ColSpace;
  using RhsInner = typename spaces_of<Rhs>::RowSpace;
  static constexpr bool value =
      std::disjunction<std::bool_constant<is_dimensionless_space<LhsInner> && is_dimensionless_space<RhsInner>>,
                       spaces_dual<LhsInner, RhsInner>>::value;
};

template <typename BinaryOp, typename Lhs, typename Rhs>
struct binary_op_spaces {
  static constexpr unit_combination combination = binary_op_unit_combination<remove_all_t<BinaryOp>>::value;
  static constexpr bool value = combination == unit_combination::equal ? xpr_spaces_equal<Lhs, Rhs>::value
                                : combination == unit_combination::product
                                    ? !has_typed_spaces<Lhs> || !has_typed_spaces<Rhs>
                                    : !has_typed_spaces<Rhs>;
  // The typed operand's spaces: the left one's unless only the right one is typed.
  using Typed = std::conditional_t<combination == unit_combination::product && !has_typed_spaces<Lhs>, Rhs, Lhs>;
  using RowSpace = typename spaces_of<Typed>::RowSpace;
  using ColSpace = typename spaces_of<Typed>::ColSpace;
};

template <typename Func, typename Dst, typename Src>
struct assignment_spaces {
  static constexpr bool value = assign_op_unit_combination<Func>::value == unit_combination::equal
                                    ? xpr_spaces_equal<Dst, Src>::value
                                    : !has_typed_spaces<Src>;
};

// Not for the Transpose of a TranspositionsBase or PermutationBase, which has no traits of its own; the fallback reads
// the Transpose's.
template <typename MatrixType>
  requires requires { traits<MatrixType>::RowsAtCompileTime; }
struct spaces_of<Transpose<MatrixType>> {
  using RowSpace = typename spaces_of<MatrixType>::ColSpace;
  using ColSpace = typename spaces_of<MatrixType>::RowSpace;
};

template <typename Lhs, typename Rhs, int Option>
struct spaces_of<Product<Lhs, Rhs, Option>> {
  using RowSpace = typename spaces_of<Lhs>::RowSpace;
  using ColSpace = typename spaces_of<Rhs>::ColSpace;
};

template <typename BinaryOp, typename Lhs, typename Rhs>
struct spaces_of<CwiseBinaryOp<BinaryOp, Lhs, Rhs>> {
  using RowSpace = typename binary_op_spaces<BinaryOp, remove_all_t<Lhs>, remove_all_t<Rhs>>::RowSpace;
  using ColSpace = typename binary_op_spaces<BinaryOp, remove_all_t<Lhs>, remove_all_t<Rhs>>::ColSpace;
};

// Zero(), Constant(), Identity(), and setZero() on a typed matrix are in its spaces; a scalar operand's constant is
// built on an untyped Matrix or Array (plain_constant_type), so it stays dimensionless.
template <typename NullaryOp, typename PlainObjectType>
struct spaces_of<CwiseNullaryOp<NullaryOp, PlainObjectType>> : spaces_of<PlainObjectType> {};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_RULES_H
