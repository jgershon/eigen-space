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
  using type =
      std::conditional_t<N == 0, RunList<UnitRun<V, M>, Rest...>,
                         std::conditional_t<alikeQuantities(U, V), RunList<UnitRun<U, addRunSizes(N, M)>, Rest...>,
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
  using type = UnitRun<inverseQuantity(U), N>;
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

/** \internal The run list of the leaves \a Leaves (a std::tuple of LeafRun and RepeatedLeaves): adjacent leaves of one
 * quantity merged, empty ones dropped, and a repeated block as one resizable run where it has one quantity.
 */
template <typename Leaves>
struct RunsOfLeaves;

template <typename Leaf>
struct RunsOfLeaf;

template <auto Q, int N>
struct RunsOfLeaf<LeafRun<Q, N>> {
  using type = typename PrependRun<UnitRun<Q, N>, RunList<>>::type;
};

template <typename Inner>
struct RunsOfLeaf<RepeatedLeaves<Inner>> {
  using InnerRuns = typename RunsOfLeaves<Inner>::type;
  template <typename R>
  struct Repeat {
    using type = RunList<RepeatedRuns<R>>;
  };
  template <auto Q, int N>
  struct Repeat<RunList<UnitRun<Q, N>>> {
    using type = RunList<UnitRun<Q, Dynamic>>;
  };
  using type = typename Repeat<InnerRuns>::type;
};

template <>
struct RunsOfLeaves<std::tuple<>> {
  using type = RunList<>;
};

template <typename First, typename... Rest>
struct RunsOfLeaves<std::tuple<First, Rest...>> {
  using type =
      typename ConcatRuns<typename RunsOfLeaf<First>::type, typename RunsOfLeaves<std::tuple<Rest...>>::type>::type;
};

/** \internal The run list of the space \a S. */
template <typename S>
using SpaceRuns = typename RunsOfLeaves<typename SpaceLeaves<S>::type>::type;

/** \internal Coordinates paired across two run lists: \a N of them (\c Dynamic if resizable), measuring \a Qa in one
 * and \a Qb in the other. */
template <auto Qa, auto Qb, int N>
struct RunPair {};

/** \internal Repeated blocks paired across two run lists, with the pairs of one repetition. */
template <typename Pairs>
struct RepeatedPair {};

/** \internal The pairs of two run lists refined to one partition. */
template <typename... Pairs>
struct RunPairs {};

/** \internal Two run lists that have no common partition: sizes that cannot agree, a resizable run before the last,
 * or a repeated block against a run. */
struct RefineFail {};

/** \internal The run lists \a A and \a B split at each other's boundaries into pairs of runs of one size, so that two
 * spaces whose blocks fall differently (metres as 3 and 2 coordinates against metres as 5) compare coordinate by
 * coordinate. A resizable run pairs only as the last of both lists, whose total size is checked at run time. */
template <typename A, typename B, typename Acc = RunPairs<>>
struct Refine {
  using type = RefineFail;
};

template <typename... P>
struct Refine<RunList<>, RunList<>, RunPairs<P...>> {
  using type = RunPairs<P...>;
};

struct RefineFailed {
  using type = RefineFail;
};

template <auto U, int N, typename... As, auto V, int M, typename... Bs, typename... P>
struct Refine<RunList<UnitRun<U, N>, As...>, RunList<UnitRun<V, M>, Bs...>, RunPairs<P...>> {
  static constexpr bool resizable = N == Dynamic || M == Dynamic;
  static constexpr bool last = sizeof...(As) == 0 && sizeof...(Bs) == 0;
  using type = typename std::conditional_t<
      N == M || (resizable && last),
      Refine<RunList<As...>, RunList<Bs...>, RunPairs<P..., RunPair<U, V, (N == Dynamic ? M : N)>>>,
      std::conditional_t<
          resizable, RefineFailed,
          std::conditional_t<
              (N < M), Refine<RunList<As...>, RunList<UnitRun<V, M - N>, Bs...>, RunPairs<P..., RunPair<U, V, N>>>,
              Refine<RunList<UnitRun<U, N - M>, As...>, RunList<Bs...>, RunPairs<P..., RunPair<U, V, M>>>>>>::type;
};

template <typename InnerA, typename... As, typename InnerB, typename... Bs, typename... P>
struct Refine<RunList<RepeatedRuns<InnerA>, As...>, RunList<RepeatedRuns<InnerB>, Bs...>, RunPairs<P...>> {
  using Inner = typename Refine<InnerA, InnerB>::type;
  using type =
      typename std::conditional_t<std::is_same_v<Inner, RefineFail>, RefineFailed,
                                  Refine<RunList<As...>, RunList<Bs...>, RunPairs<P..., RepeatedPair<Inner>>>>::type;
};

/** \internal The pairs of the run lists of the spaces \a A and \a B. */
template <typename A, typename B>
using SpacePairs = typename Refine<SpaceRuns<A>, SpaceRuns<B>>::type;

/** \internal Whether \a Pred holds of every pair in \a Pairs, and \a Pairs refined at all. */
template <template <auto, auto> class Pred, typename Pairs>
struct AllPairs : std::false_type {};

template <template <auto, auto> class Pred, typename Pair>
struct PairHolds;

template <template <auto, auto> class Pred, auto U, auto V, int N>
struct PairHolds<Pred, RunPair<U, V, N>> : Pred<U, V> {};

template <template <auto, auto> class Pred, typename Inner>
struct PairHolds<Pred, RepeatedPair<Inner>> : AllPairs<Pred, Inner> {};

template <template <auto, auto> class Pred, typename... P>
struct AllPairs<Pred, RunPairs<P...>> : std::bool_constant<(PairHolds<Pred, P>::value && ...)> {};

template <auto U, auto V>
struct CompatiblePred : std::bool_constant<compatibleQuantities(U, V)> {};
template <auto U, auto V>
struct AlikePred : std::bool_constant<alikeQuantities(U, V)> {};
template <auto U, auto V>
struct AssignablePred : std::bool_constant<assignableQuantity(U, V)> {};

/** \internal Whether every coordinate of \a S is dimensionless; immediate for a plain Linear, the space of every
 * untyped matrix, so that untyped code skips the rules. */
template <typename S>
inline constexpr bool is_dimensionless_space = isDimensionlessSpace<S>();

template <int N, int M>
inline constexpr bool is_dimensionless_space<QLinear<dimensionless, N, M>> = true;

/** \internal Whether \a S is plain: dimensionless and a group under addition [additive-groups], so that the rules have
 * nothing to check; immediate for a plain Linear. A vector of unit quaternions is dimensionless and not plain. */
template <typename S>
inline constexpr bool is_plain_space = isDimensionlessSpace<S>() && isAdditiveGroup<S>();

template <int N, int M>
inline constexpr bool is_plain_space<QLinear<dimensionless, N, M>> = true;

/** \internal Whether every leaf of \a S has a unit, as a matrix's spaces need; immediate for a plain Linear. */
template <typename S>
struct space_is_bound : std::bool_constant<isBoundSpace<S>()> {};

/** \internal Whether every coordinate of \a S holds a \a Scalar of the right character [complex-scalars]. */
template <typename S, typename Scalar>
struct space_holds_scalar : std::bool_constant<spaceHoldsScalar<S, Scalar>()> {};

/** \internal Whether the spaces \a A and \a B measure compatible quantities, coordinate by coordinate: the rule of a
 * sum, converting where the units differ. */
template <typename A, typename B>
struct spaces_equal : std::disjunction<std::is_same<A, B>, AllPairs<CompatiblePred, SpacePairs<A, B>>> {};

/** \internal Whether the spaces \a A and \a B measure the same quantities in equivalent units, so that nothing
 * converts between them. */
template <typename A, typename B>
struct spaces_alike : std::disjunction<std::is_same<A, B>, AllPairs<AlikePred, SpacePairs<A, B>>> {};

/** \internal Whether \a A is the dual of \a B up to a conversion, coordinate by coordinate. */
template <typename A, typename B>
struct spaces_dual
    : AllPairs<CompatiblePred, typename Refine<SpaceRuns<A>, typename InvertRuns<SpaceRuns<B>>::type>::type> {};

/** \internal The space of the runs \a Runs, untagged: a QLinear for each run, in a Composite where there are several.
 */
template <typename Runs>
struct SpaceOfRuns;

template <typename Run>
struct LeafOfRun;

template <auto Q, int N>
struct LeafOfRun<UnitRun<Q, N>> {
  using type = QLinear<Q, N, N>;
};

template <typename Inner>
struct LeafOfRun<RepeatedRuns<Inner>> {
  using type = Repeated<typename SpaceOfRuns<Inner>::type>;
};

template <>
struct SpaceOfRuns<RunList<>> {
  using type = Linear<0>;
};

template <typename Run>
struct SpaceOfRuns<RunList<Run>> {
  using type = typename LeafOfRun<Run>::type;
};

template <typename First, typename Second, typename... Rest>
struct SpaceOfRuns<RunList<First, Second, Rest...>> {
  using type =
      Composite<typename LeafOfRun<First>::type, typename LeafOfRun<Second>::type, typename LeafOfRun<Rest>::type...>;
};

/** \internal The common runs of the pairs \a Pairs: each pair's common quantity [heterogeneous-results]. */
template <typename Pairs>
struct CommonRunsOf;

template <typename Pair>
struct CommonRunOf;

template <auto U, auto V, int N>
struct CommonRunOf<RunPair<U, V, N>> {
  using type = typename PrependRun<UnitRun<commonQuantity(U, V), N>, RunList<>>::type;
};

template <typename Inner>
struct CommonRunOf<RepeatedPair<Inner>> {
  using InnerRuns = typename CommonRunsOf<Inner>::type;
  template <typename R>
  struct Repeat {
    using type = RunList<RepeatedRuns<R>>;
  };
  template <auto Q, int N>
  struct Repeat<RunList<UnitRun<Q, N>>> {
    using type = RunList<UnitRun<Q, Dynamic>>;
  };
  using type = typename Repeat<InnerRuns>::type;
};

template <>
struct CommonRunsOf<RunPairs<>> {
  using type = RunList<>;
};

template <typename First, typename... Rest>
struct CommonRunsOf<RunPairs<First, Rest...>> {
  using type =
      typename ConcatRuns<typename CommonRunOf<First>::type, typename CommonRunsOf<RunPairs<Rest...>>::type>::type;
};

/** \internal \a S with its bindings taken off, keeping its tags and structure: what two spaces in different units of
 * the same tags have in common. */
template <typename S>
struct UnbindOf {
  using type = S;
};

template <typename S>
  requires IsBoundSpace<S>
struct UnbindOf<S> {
  using type = typename UnbindOf<BoundBase<S>>::type;
};

template <typename... Blocks>
struct UnbindOf<Composite<Blocks...>> {
  using type = Composite<typename UnbindOf<Blocks>::type...>;
};

template <typename Block, int MaxCount>
struct UnbindOf<Repeated<Block, MaxCount>> {
  using type = Repeated<typename UnbindOf<Block>::type, MaxCount>;
};

/** \internal \a S with the quantities \a Qs (a tuple of QuantityValue) bound to its leaves. */
template <typename S, typename Qs>
struct BindSlots;

template <typename S, typename... Qs>
struct BindSlots<S, std::tuple<Qs...>> {
  using type = BoundSpace<S, Qs::value...>;
};

template <typename QsA, typename QsB>
struct CommonSlots;

template <typename... A, typename... B>
struct CommonSlots<std::tuple<A...>, std::tuple<B...>> {
  using type = std::tuple<QuantityValue<commonQuantity(A::value, B::value)>...>;
};

/** \internal The space a sum of operands in \a A and \a B is in [heterogeneous-results]: \a A where the two are one
 * type; their tags rebound to the common units where they share tags; and otherwise the untagged leaves of the
 * common quantities. It does not depend on the order of \a A and \a B. */
template <typename A, typename B>
struct CommonSpaceOf;

template <typename A, typename B>
struct CommonSpaceByRuns {
  using type = typename SpaceOfRuns<typename CommonRunsOf<SpacePairs<A, B>>::type>::type;
};

template <typename A, typename B>
struct CommonSpaceByTags {
  using type = typename BindSlots<typename UnbindOf<A>::type,
                                  typename CommonSlots<SlotQuantities<A>, SlotQuantities<B>>::type>::type;
};

template <typename A, typename B>
struct CommonDualSpace {
  using type = DualSpace<typename CommonSpaceOf<DualInner<A>, DualInner<B>>::type>;
};

template <typename A, typename B>
struct CommonSpaceOf {
  using type = typename std::conditional_t<
      std::is_same_v<A, B>, std::type_identity<A>,
      std::conditional_t<IsDualSpace<A> && IsDualSpace<B> && !IsBoundSpace<A> && !IsBoundSpace<B>,
                         CommonDualSpace<A, B>,
                         std::conditional_t<std::is_same_v<typename UnbindOf<A>::type, typename UnbindOf<B>::type>,
                                            CommonSpaceByTags<A, B>, CommonSpaceByRuns<A, B>>>>::type;
};

/** \internal \a S with every quantity times \a C: the row space of a product whose inner coordinates multiply to
 * \a C [heterogeneous-results]. A ratio of units keeps the tags; a quantity with a dimension changes what the leaves
 * measure, so the result is untagged. */
template <typename S, auto C>
struct ScaledSpaceOf;

template <typename Qs, auto C>
struct ScaledSlots;

template <typename... Qs, auto C>
struct ScaledSlots<std::tuple<Qs...>, C> {
  using type = std::tuple<QuantityValue<quantityProduct(Qs::value, C)>...>;
};

template <typename Runs, auto C>
struct ScaledRuns;

template <typename Run, auto C>
struct ScaledRun;

template <auto Q, int N, auto C>
struct ScaledRun<UnitRun<Q, N>, C> {
  using type = UnitRun<quantityProduct(Q, C), N>;
};

template <typename Inner, auto C>
struct ScaledRun<RepeatedRuns<Inner>, C> {
  using type = RepeatedRuns<typename ScaledRuns<Inner, C>::type>;
};

template <typename... Runs, auto C>
struct ScaledRuns<RunList<Runs...>, C> {
  using type = RunList<typename ScaledRun<Runs, C>::type...>;
};

template <typename S, auto C>
struct ScaledByTags {
  using type = typename BindSlots<typename UnbindOf<S>::type, typename ScaledSlots<SlotQuantities<S>, C>::type>::type;
};

template <typename S, auto C>
struct ScaledByRuns {
  using type = typename SpaceOfRuns<typename ScaledRuns<SpaceRuns<S>, C>::type>::type;
};

template <typename S, auto C>
struct ScaledDual {
  using type = DualSpace<typename ScaledSpaceOf<DualInner<S>, inverseQuantity(C)>::type>;
};

template <typename S, auto C>
struct ScaledSpaceOf {
  using type = typename std::conditional_t<
      isDimensionless(C), std::type_identity<S>,
      std::conditional_t<IsDualSpace<S> && !IsBoundSpace<S>, ScaledDual<S, C>,
                         std::conditional_t<isDimensionlessKind(C), ScaledByTags<S, C>, ScaledByRuns<S, C>>>>::type;
};

/** \internal Whether an expression of type \a T has a unit anywhere. An expression that spaces_of is not
 * specialized for has dimensionless Linear spaces, so it has none, and meets the rules only with untyped operands. */
template <typename T>
inline constexpr bool has_typed_spaces = !(is_dimensionless_space<typename spaces_of<T>::RowSpace> &&
                                           is_dimensionless_space<typename spaces_of<T>::ColSpace>);

/** \internal Whether an expression of type \a T has spaces the rules check: a unit, or a space that is not a group
 * under addition. */
template <typename T>
inline constexpr bool has_ruled_spaces =
    !(is_plain_space<typename spaces_of<T>::RowSpace> && is_plain_space<typename spaces_of<T>::ColSpace>);

/** \internal Whether both spaces of an expression of type \a T are groups under addition [additive-groups]. */
template <typename T>
inline constexpr bool has_additive_spaces =
    isAdditiveGroup<typename spaces_of<T>::RowSpace>() && isAdditiveGroup<typename spaces_of<T>::ColSpace>();

/** \internal The rule of a sum or difference of expressions in the spaces of \a A and \a B: whether they are
 * compatible, the common spaces of the result, and whether an operand converts into them. Immediate where neither
 * operand has a unit. */
template <typename A, typename B, bool Untyped = !has_ruled_spaces<A> && !has_ruled_spaces<B>>
struct sum_spaces {
  static constexpr bool value = true;
  static constexpr bool converts = false;
  using RowSpace = typename spaces_of<A>::RowSpace;
  using ColSpace = typename spaces_of<A>::ColSpace;
};

template <typename A, typename B>
struct sum_spaces<A, B, false> {
  using AR = typename spaces_of<A>::RowSpace;
  using AC = typename spaces_of<A>::ColSpace;
  using BR = typename spaces_of<B>::RowSpace;
  using BC = typename spaces_of<B>::ColSpace;
  static constexpr bool additive = has_additive_spaces<A> && has_additive_spaces<B>;
  static constexpr bool value = additive && spaces_equal<AR, BR>::value && spaces_equal<AC, BC>::value;
  using RowSpace = typename std::conditional_t<value, CommonSpaceOf<AR, BR>, std::type_identity<AR>>::type;
  using ColSpace = typename std::conditional_t<value, CommonSpaceOf<AC, BC>, std::type_identity<AC>>::type;
  static constexpr bool converts = value && !(spaces_alike<AR, RowSpace>::value && spaces_alike<BR, RowSpace>::value &&
                                              spaces_alike<AC, ColSpace>::value && spaces_alike<BC, ColSpace>::value);
};

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

template <typename BinaryOp, typename Lhs, typename Rhs>
struct binary_op_spaces {
  static constexpr unit_combination combination = binary_op_unit_combination<remove_all_t<BinaryOp>>::value;
  // The typed operand's spaces: the left one's unless only the right one is typed.
  using Typed = std::conditional_t<combination == unit_combination::product && !has_typed_spaces<Lhs>, Rhs, Lhs>;
  struct Scaling {
    static constexpr bool value =
        (combination == unit_combination::product ? !has_typed_spaces<Lhs> || !has_typed_spaces<Rhs>
                                                  : !has_typed_spaces<Rhs>) &&
        has_additive_spaces<Lhs> && has_additive_spaces<Rhs>;
    static constexpr bool converts = false;
    using RowSpace = typename spaces_of<Typed>::RowSpace;
    using ColSpace = typename spaces_of<Typed>::ColSpace;
  };
  using Rule = std::conditional_t<combination == unit_combination::equal, sum_spaces<Lhs, Rhs>, Scaling>;
  static constexpr bool value = Rule::value;
  static constexpr bool converts = Rule::converts;
  using RowSpace = typename Rule::RowSpace;
  using ColSpace = typename Rule::ColSpace;
};

template <typename Func>
struct is_accumulating_assignment : std::false_type {};
template <typename A, typename B>
struct is_accumulating_assignment<add_assign_op<A, B>> : std::true_type {};
template <typename A, typename B>
struct is_accumulating_assignment<sub_assign_op<A, B>> : std::true_type {};

template <typename Func, typename Dst, typename Src, bool Untyped = !has_ruled_spaces<Dst> && !has_ruled_spaces<Src>>
struct assignment_spaces_impl {
  static constexpr bool value = true;
  static constexpr bool converts = false;
};

template <typename Func, typename Dst, typename Src>
struct assignment_spaces_impl<Func, Dst, Src, false> {
  using DR = typename spaces_of<Dst>::RowSpace;
  using DC = typename spaces_of<Dst>::ColSpace;
  using SR = typename spaces_of<Src>::RowSpace;
  using SC = typename spaces_of<Src>::ColSpace;
  static constexpr bool equal = assign_op_unit_combination<Func>::value == unit_combination::equal;
  // A copy needs no group; +=, -=, *=, and /= act on coordinates, so they need one under addition [additive-groups].
  static constexpr bool additive = !(is_accumulating_assignment<Func>::value || !equal) || has_additive_spaces<Dst>;
  static constexpr bool value =
      additive &&
      (equal ? std::disjunction<std::is_same<DR, SR>, AllPairs<AssignablePred, SpacePairs<SR, DR>>>::value &&
                   std::disjunction<std::is_same<DC, SC>, AllPairs<AssignablePred, SpacePairs<SC, DC>>>::value
             : !has_typed_spaces<Src>);
  static constexpr bool converts = equal && value && !(spaces_alike<SR, DR>::value && spaces_alike<SC, DC>::value);
};

template <typename Func, typename Dst, typename Src>
struct assignment_spaces : assignment_spaces_impl<Func, Dst, Src> {};

/** \internal The quantities the inner coordinates of a product multiply to, pair by pair, as a tuple of
 * QuantityValue. */
template <typename Pairs>
struct InnerProducts;

template <typename Pair>
struct InnerProductsOf;

template <auto U, auto V, int N>
struct InnerProductsOf<RunPair<U, V, N>> {
  using type = std::tuple<QuantityValue<quantityProduct(U, V)>>;
};

template <typename Inner>
struct InnerProductsOf<RepeatedPair<Inner>> : InnerProducts<Inner> {};

template <typename... P>
struct InnerProducts<RunPairs<P...>> {
  using type = decltype(std::tuple_cat(std::declval<typename InnerProductsOf<P>::type>()...));
};

template <auto First, auto... Rest>
consteval auto commonOfAll() {
  if constexpr (sizeof...(Rest) == 0) {
    return First;
  } else {
    return commonQuantity(First, commonOfAll<Rest...>());
  }
}

template <typename Cs>
struct InnerCommon;

template <>
struct InnerCommon<std::tuple<>> {
  static constexpr bool compatible = true;
  static constexpr bool alike = true;
  static constexpr auto value = dimensionless;
};

template <typename First, typename... Rest>
struct InnerCommon<std::tuple<First, Rest...>> {
  static constexpr bool compatible = (compatibleQuantities(First::value, Rest::value) && ...);
  static constexpr bool alike = (alikeQuantities(First::value, Rest::value) && ...);
  static constexpr auto value = commonOfAll<First::value, Rest::value...>();
};

/** \internal The rule of a product of \a Lhs and \a Rhs [heterogeneous-results]: the inner coordinates,
 * u_C_lhs(k) u_R_rhs(k), share one dimension; where they differ in unit, the right operand's rows convert to their
 * common unit c, which operator* does; and the result is in (R_lhs c, C_rhs). Immediate where neither inner space has
 * a unit. */
template <typename Lhs, typename Rhs, typename LhsInner = typename spaces_of<Lhs>::ColSpace,
          typename RhsInner = typename spaces_of<Rhs>::RowSpace,
          bool Untyped = is_plain_space<LhsInner> && is_plain_space<RhsInner>>
struct product_spaces_impl {
  static constexpr bool value = true;
  static constexpr bool converts = false;
  using RowSpace = typename spaces_of<Lhs>::RowSpace;
  using ColSpace = typename spaces_of<Rhs>::ColSpace;
};

template <typename Lhs, typename Rhs, typename LhsInner, typename RhsInner>
struct product_spaces_impl<Lhs, Rhs, LhsInner, RhsInner, false> {
  using Pairs = SpacePairs<LhsInner, RhsInner>;
  struct NoPairs {
    static constexpr bool compatible = false;
    static constexpr bool alike = false;
    static constexpr auto value = dimensionless;
  };
  template <typename P>
  struct CommonOf : InnerCommon<typename InnerProducts<P>::type> {};
  using Common = typename std::conditional_t<std::is_same_v<Pairs, RefineFail>, std::type_identity<NoPairs>,
                                             std::type_identity<CommonOf<Pairs>>>::type;
  static constexpr bool value =
      Common::compatible && isAdditiveGroup<LhsInner>() && isAdditiveGroup<RhsInner>();  // [additive-groups]
  static constexpr bool converts = value && !Common::alike;
  using RowSpace = typename std::conditional_t<value, ScaledSpaceOf<typename spaces_of<Lhs>::RowSpace, Common::value>,
                                               std::type_identity<typename spaces_of<Lhs>::RowSpace>>::type;
  using ColSpace = typename spaces_of<Rhs>::ColSpace;
};

template <typename Lhs, typename Rhs>
struct product_spaces : product_spaces_impl<Lhs, Rhs> {};

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
  using RowSpace = typename product_spaces<Lhs, Rhs>::RowSpace;
  using ColSpace = typename product_spaces<Lhs, Rhs>::ColSpace;
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

/** \internal The identity on coefficients, reporting the spaces \a RowSpace and \a ColSpace: the last step of a
 * conversion, whose values are already in those spaces' units. */
template <typename RowSpace, typename ColSpace, typename Scalar>
struct space_relabel_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar& operator()(const Scalar& a) const { return a; }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet& packetOp(const Packet& a) const {
    return a;
  }
};

template <typename RowSpace, typename ColSpace, typename Scalar>
struct functor_traits<space_relabel_op<RowSpace, ColSpace, Scalar>> {
  enum { Cost = 0, PacketAccess = true };
};

template <typename RowSpace_, typename ColSpace_, typename Scalar, typename Xpr>
struct spaces_of<CwiseUnaryOp<space_relabel_op<RowSpace_, ColSpace_, Scalar>, Xpr>> {
  using RowSpace = RowSpace_;
  using ColSpace = ColSpace_;
};

/** \internal Whether a coefficient-wise unary operation keeps what its operand measures: negation and the complex
 * parts and conjugate, so that adjoint() swaps the spaces as transpose() does [complex-scalars]. Any other unary
 * operation has dimensionless spaces, as an expression the rules do not reach. */
template <typename Op>
struct unary_op_keeps_spaces : std::false_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_opposite_op<S>> : std::true_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_conjugate_op<S>> : std::true_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_real_op<S>> : std::true_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_imag_op<S>> : std::true_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_abs_op<S>> : std::true_type {};

template <typename S>
struct unary_op_keeps_spaces<scalar_real_ref_op<S>> : std::true_type {};
template <typename S>
struct unary_op_keeps_spaces<scalar_imag_ref_op<S>> : std::true_type {};

template <typename UnaryOp, typename Xpr>
  requires(unary_op_keeps_spaces<UnaryOp>::value)
struct spaces_of<CwiseUnaryOp<UnaryOp, Xpr>> : spaces_of<remove_all_t<Xpr>> {};

// The writable real() and imag() of a complex matrix.
template <typename ViewOp, typename Xpr, typename StrideType>
  requires(unary_op_keeps_spaces<ViewOp>::value)
struct spaces_of<CwiseUnaryView<ViewOp, Xpr, StrideType>> : spaces_of<remove_all_t<Xpr>> {};

/** \internal \a xpr reported in the spaces \a ToRow and \a ToCol, with its coefficients as they are. */
template <typename ToRow, typename ToCol, typename Xpr>
auto relabelSpaces(const Xpr& xpr) {
  return CwiseUnaryOp<space_relabel_op<ToRow, ToCol, typename traits<Xpr>::Scalar>, const Xpr>(xpr);
}

/** \internal The conversion factor of each run of an axis, from the pairs \a Pairs of its source and target runs. */
template <typename Pairs>
struct AxisFactors;

template <typename Pair>
struct PairFactor;

template <auto U, auto V, int N>
struct PairFactor<RunPair<U, V, N>> {
  static constexpr int size = N;
  static constexpr double factor = conversionFactor(U, V);
};

template <typename... P>
struct AxisFactors<RunPairs<P...>> {
  static constexpr int count = int(sizeof...(P));
  static constexpr std::array<int, sizeof...(P)> sizes{PairFactor<P>::size...};
  static constexpr std::array<double, sizeof...(P)> factors{PairFactor<P>::factor...};

  /** The factor of coordinate \a i of an axis of \a n coordinates, whose one resizable run takes what the others
   * leave. */
  static double at(Index i, Index n) {
    Index fixed = 0;
    for (int k = 0; k < count; ++k) fixed += sizes[k] == Dynamic ? 0 : sizes[k];
    Index start = 0;
    for (int k = 0; k < count; ++k) {
      const Index size = sizes[k] == Dynamic ? n - fixed : sizes[k];
      if (i < start + size) return factors[k];
      start += size;
    }
    return 1.0;
  }
};

/** \internal The factor that converts coefficient (i, j) of a matrix: its row's times its column's. */
template <typename Scalar, typename RowFactors, typename ColFactors>
struct space_factor_op {
  Index rows = 0;
  Index cols = 0;
  template <typename IndexType>
  EIGEN_STRONG_INLINE Scalar operator()(IndexType i, IndexType j) const {
    return Scalar(RowFactors::at(Index(i), rows) * ColFactors::at(Index(j), cols));
  }
};

template <typename Scalar, typename RowFactors, typename ColFactors>
struct functor_traits<space_factor_op<Scalar, RowFactors, ColFactors>> {
  enum { Cost = 10, PacketAccess = false, IsRepeatable = true };
};

template <typename Pairs>
struct HasRepeatedPair : std::false_type {};
template <typename... P>
struct HasRepeatedPair<RunPairs<P...>> : std::disjunction<HasRepeatedPair<P>...> {};
template <typename Inner>
struct HasRepeatedPair<RepeatedPair<Inner>> : std::true_type {};

/** \internal \a xpr in the spaces \a ToRow and \a ToCol [heterogeneous-results]: each coefficient times the factors
 * of its row and column, then relabeled; only relabeled where the units already agree. */
template <typename ToRow, typename ToCol, typename Xpr>
auto convertSpaces(const Xpr& xpr) {
  using FromRow = typename spaces_of<Xpr>::RowSpace;
  using FromCol = typename spaces_of<Xpr>::ColSpace;
  using RowPairs = SpacePairs<FromRow, ToRow>;
  using ColPairs = SpacePairs<FromCol, ToCol>;
  static_assert(spaces_equal<FromRow, ToRow>::value && spaces_equal<FromCol, ToCol>::value,
                "a matrix converts only into spaces that measure compatible quantities");
  using Scalar = typename traits<Xpr>::Scalar;
  using Relabel = space_relabel_op<ToRow, ToCol, Scalar>;
  if constexpr (spaces_alike<FromRow, ToRow>::value && spaces_alike<FromCol, ToCol>::value) {
    return CwiseUnaryOp<Relabel, const Xpr>(xpr);
  } else {
    static_assert(!HasRepeatedPair<RowPairs>::value && !HasRepeatedPair<ColPairs>::value,
                  "a conversion across a repeated block of several units is not supported yet");
    using Factors = space_factor_op<Scalar, AxisFactors<RowPairs>, AxisFactors<ColPairs>>;
    using Plain = std::conditional_t<std::is_same_v<typename traits<Xpr>::XprKind, MatrixXpr>,
                                     typename plain_constant_type<Xpr, Scalar>::matrix_type,
                                     typename plain_constant_type<Xpr, Scalar>::array_type>;
    using FactorXpr = CwiseNullaryOp<Factors, const Plain>;
    using Scaled = CwiseBinaryOp<scalar_product_op<Scalar, Scalar>, const Xpr, const FactorXpr>;
    return CwiseUnaryOp<Relabel, const Scaled>(
        Scaled(xpr, FactorXpr(xpr.rows(), xpr.cols(), Factors{xpr.rows(), xpr.cols()})));
  }
}

/** \internal The right operand of a product converted so that its inner coordinates multiply with the left
 * operand's to one unit c: run k of its rows measures c / u_C_lhs(k). */
template <typename Lhs, typename Rhs>
struct product_rhs_conversion {
  using Spaces = product_spaces<Lhs, Rhs>;
  using Pairs = typename Spaces::Pairs;
  static_assert(!HasRepeatedPair<Pairs>::value,
                "a product converting across a repeated block of several units is not supported yet");
  template <typename P>
  struct RowRuns;
  template <typename... P>
  struct RowRuns<RunPairs<P...>> {
    template <typename Pair>
    struct Of;
    template <auto U, auto V, int N>
    struct Of<RunPair<U, V, N>> {
      using type =
          typename PrependRun<UnitRun<quantityProduct(inverseQuantity(U), Spaces::Common::value), N>, RunList<>>::type;
    };
    template <typename Pending, typename... Rest>
    struct Fold {
      using type = Pending;
    };
    template <typename Pending, typename First, typename... Rest>
    struct Fold<Pending, First, Rest...> {
      using type = typename Fold<typename ConcatRuns<Pending, typename Of<First>::type>::type, Rest...>::type;
    };
    using type = typename Fold<RunList<>, P...>::type;
  };
  using RowSpace = typename SpaceOfRuns<typename RowRuns<Pairs>::type>::type;
  using ColSpace = typename spaces_of<Rhs>::ColSpace;
  using type = decltype(convertSpaces<RowSpace, ColSpace>(std::declval<const Rhs&>()));
  static type convert(const Rhs& rhs) { return convertSpaces<RowSpace, ColSpace>(rhs); }
};

/** \internal A sum or difference whose operands convert into its common spaces, with the conversions applied. */
template <typename BinaryOp, typename Lhs, typename Rhs>
struct converting_binary {
  using Spaces = binary_op_spaces<BinaryOp, remove_all_t<Lhs>, remove_all_t<Rhs>>;
  using RowSpace = typename Spaces::RowSpace;
  using ColSpace = typename Spaces::ColSpace;
  using ConvertedLhs = decltype(convertSpaces<RowSpace, ColSpace>(std::declval<const remove_all_t<Lhs>&>()));
  using ConvertedRhs = decltype(convertSpaces<RowSpace, ColSpace>(std::declval<const remove_all_t<Rhs>&>()));
  using type = CwiseBinaryOp<BinaryOp, const ConvertedLhs, const ConvertedRhs>;
  static type make(const CwiseBinaryOp<BinaryOp, Lhs, Rhs>& xpr) {
    return type(convertSpaces<RowSpace, ColSpace>(xpr.lhs()), convertSpaces<RowSpace, ColSpace>(xpr.rhs()),
                xpr.functor());
  }
};

/** \internal The evaluator of a sum or difference whose operands differ in units: that of the same operation on the
 * operands converted [heterogeneous-results]. The converted expression is a temporary; its evaluator keeps the
 * operands' evaluators and the functors, which do not refer back to it. */
template <typename BinaryOp, typename Lhs, typename Rhs>
  requires(binary_op_spaces<BinaryOp, remove_all_t<Lhs>, remove_all_t<Rhs>>::converts)
struct evaluator<CwiseBinaryOp<BinaryOp, Lhs, Rhs>> : evaluator<typename converting_binary<BinaryOp, Lhs, Rhs>::type> {
  using XprType = CwiseBinaryOp<BinaryOp, Lhs, Rhs>;
  using Base = evaluator<typename converting_binary<BinaryOp, Lhs, Rhs>::type>;
  EIGEN_DEVICE_FUNC explicit evaluator(const XprType& xpr) : Base(converting_binary<BinaryOp, Lhs, Rhs>::make(xpr)) {}
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_RULES_H
