// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_SPACES_COMPOSITE_H
#define EIGEN_SPACES_COMPOSITE_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/** \internal The difference space of \a Block as a block of a Composite or Repeated: its own difference space, wrapped
 * in DifferenceOf where that would lose \a Block's tag (Spaces/TagPath.h).
 */
template <typename Block>
struct DifferenceBlockOf;

/** \internal How PartStorage keeps \a Part: in a one-element tuple if it has state, else not at all. */
template <typename Part>
using StoredPart = std::conditional_t<std::is_empty_v<Part>, std::tuple<>, std::tuple<Part>>;

/** \internal The parts with state, held as a member so that \c std is not an associated namespace. */
template <typename Stored>
struct StoredParts {
  Stored parts;  ///< the parts with state, in order

  friend bool operator==(const StoredParts&, const StoredParts&) = default;
};

/** \internal No parts with state: an empty base. */
template <>
struct StoredParts<std::tuple<>> {
  StoredParts() = default;

  /** Accepts the empty tuple that PartStorage builds when no part has state. */
  explicit StoredParts(std::tuple<>) {}

  friend bool operator==(const StoredParts&, const StoredParts&) = default;
};

/** \internal The parts of a Composite (its blocks), in order.
 *
 * Only parts with state are stored. An empty part holds nothing, so it is one shared instance,
 * and the stored parts are a base class, so that when every part is empty the storage is an
 * empty type, and so is a class derived from it (design/decisions.md [space-storage]).
 */
template <typename... Parts>
class PartStorage : private StoredParts<decltype(std::tuple_cat(std::declval<StoredPart<Parts>>()...))> {
  using Stored = decltype(std::tuple_cat(std::declval<StoredPart<Parts>>()...));
  using Base = StoredParts<Stored>;

  template <std::size_t Idx>
  using Part = std::tuple_element_t<Idx, std::tuple<Parts...>>;

  /** \returns the position of part \a I among the stored parts */
  static consteval std::size_t storedIndex(std::size_t i) {
    constexpr auto empty = std::array<bool, sizeof...(Parts)>{std::is_empty_v<Parts>...};
    std::size_t result = 0;
    for (std::size_t j = 0; j < i; ++j) {
      result += empty[j] ? 0 : 1;
    }
    return result;
  }

  /** \returns \a part in a one-element tuple if it has state, else an empty tuple */
  template <typename P>
  static StoredPart<P> keep(P& part) {
    if constexpr (std::is_empty_v<P>) {
      return {};
    } else {
      return StoredPart<P>{std::move(part)};
    }
  }

  /** The one shared instance of an empty part \a I. */
  template <std::size_t Idx>
  static inline Part<Idx> instance{};

 public:
  PartStorage() = default;

  /** Stores those of \a parts that have state. */
  explicit PartStorage(Parts... parts) : Base{std::tuple_cat(keep(parts)...)} {}

  /** \returns part \a I: the stored one, or the shared instance of an empty type */
  template <std::size_t Idx>
  const Part<Idx>& part() const {
    if constexpr (std::is_empty_v<Part<Idx>>) {
      return instance<Idx>;
    } else {
      return std::get<storedIndex(Idx)>(Base::parts);
    }
  }

  /** \returns part \a I, to change it: the stored one, or the shared instance of an empty type */
  template <std::size_t Idx>
  Part<Idx>& part() {
    if constexpr (std::is_empty_v<Part<Idx>>) {
      return instance<Idx>;
    } else {
      return std::get<storedIndex(Idx)>(Base::parts);
    }
  }

  friend bool operator==(const PartStorage&, const PartStorage&) = default;
};

}  // namespace internal

/** \ingroup Spaces_Module
 * \brief The concatenation of \a Blocks, any number of them resizable.
 *
 * Nesting a named composite names a range of coordinates; storage is always the flat
 * concatenation. Each block's offset is the total size of the blocks before it, so a block after
 * a resizable one has its offset only at run time, from idx<Path>(space). A composite of empty
 * blocks, such as any fixed-size composite of the coordinate types, is itself an empty type.
 *
 * A composite with one resizable block, at any depth, has resize(n). One with several has none,
 * since a length does not say how to split it among them: its blocks are resized through block()
 * (design/decisions.md [several-resizable-blocks]).
 */
template <Space... Blocks>
class Composite : private internal::PartStorage<Blocks...> {
  using Storage = internal::PartStorage<Blocks...>;

  static_assert(sizeof...(Blocks) > 0, "a composite needs at least one block");
  static constexpr std::size_t BlockCount = sizeof...(Blocks);
  static constexpr std::array<bool, BlockCount> Resizable{(Blocks::ValueDimensionStep != 0)...};

  /** \returns the stride of the composite: the greatest common divisor of its resizable blocks'
   * strides, or zero
   */
  static consteval int strides() {
    int result = 0;
    ((result = std::gcd(result, Blocks::ValueDimensionStep)), ...);
    return result;
  }

  /** \returns the position of the first resizable block, or \c BlockCount if none */
  static consteval std::size_t firstResizable() {
    for (std::size_t i = 0; i < BlockCount; ++i) {
      if (Resizable[i]) {
        return i;
      }
    }
    return BlockCount;
  }

 public:
  static constexpr int FixedValueDimension = (Blocks::FixedValueDimension + ...);  ///< the blocks' fixed coordinates

  /** Zero if fixed; otherwise the greatest common divisor of its resizable blocks' strides, which
   * every change in its value dimension is a multiple of.
   */
  static constexpr int ValueDimensionStep = strides();

  /** The sum of the blocks' bounds, or \c Dynamic if any is unbounded. */
  static constexpr int MaxValueDimension =
      ((Blocks::MaxValueDimension == Dynamic) || ...) ? Dynamic : (Blocks::MaxValueDimension + ...);

  Composite() = default;

  /** The concatenation of \a blocks, with their parameters and sizes. */
  explicit Composite(Blocks... blocks) : Storage{std::move(blocks)...} {}

  /** \returns block \a I */
  template <std::size_t Idx>
  const auto& block() const {
    return Storage::template part<Idx>();
  }

  /** \returns block \a I, to resize it */
  template <std::size_t Idx>
  auto& block() {
    return Storage::template part<Idx>();
  }

  /** \returns the number of coordinates: the blocks' sizes together */
  Index valueDimension() const { return offset<BlockCount>(); }

  /** \returns the offset of block \a I's first coordinate */
  template <std::size_t Idx>
  Index offset() const {
    return [&]<std::size_t... J>(std::index_sequence<J...>) {
      return (Index{0} + ... + block<J>().valueDimension());
    }(std::make_index_sequence<Idx>{});
  }

  /** Resizes the composite's only resizable block, at any depth, so the composite has \a n
   * coordinates.
   */
  void resize(Index n)
    requires((internal::resizableBlockCount<Blocks> + ...) == 1)
  {
    auto& resizable = block<firstResizable()>();
    const Index rest = valueDimension() - resizable.valueDimension();
    eigen_assert(n >= rest && "the dimension does not fit the space");
    resizable.resize(n - rest);
  }

  /** Normalizes each block's rows; a composite of blocks without normalize() has none. */
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& rows) const
    requires((internal::HasNormalize<Blocks> || ...))
  {
    forEachBlock(rows, [](const auto& block, auto&& sub) { Eigen::normalizeValues(block, sub); });
  }

  /** \f$ x \boxplus \delta \f$ block by block, for a composite with a block that has its own
   * plus(); see Eigen::plus().
   */
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const
    requires((internal::HasPlus<Blocks> || ...))
  {
    forEachBlockOf<false, true, false>(x, d, out,
                                       [](const auto& block, const auto& xs, const auto& ds, const auto& os) {
                                         internal::plusInto(block, xs, ds, os);
                                       });
  }

  /** \f$ a \boxminus b \f$ block by block, for a composite with a block that has its own
   * minus(); see Eigen::minus().
   */
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const
    requires((internal::HasMinus<Blocks> || ...))
  {
    forEachBlockOf<false, false, true>(a, b, out,
                                       [](const auto& block, const auto& as, const auto& bs, const auto& os) {
                                         internal::minusInto(block, as, bs, os);
                                       });
  }

  /** The composite of the blocks' difference spaces, with their sizes. A block whose difference space is another type
   * keeps its tag through DifferenceOf, so that a path names the same block in a covariance's rows as in a mean
   * (design/decisions.md [typed-blocks]). A composite of linear blocks is its own difference space.
   */
  auto difference() const
    requires(!(isLinear<Blocks> && ...))
  {
    return [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      return Composite<typename internal::DifferenceBlockOf<Blocks>::type...>{
          internal::DifferenceBlockOf<Blocks>::make(block<Idx>())...};
    }(std::make_index_sequence<BlockCount>{});
  }

  /** Equal when every stored parameter and size is; a fixed-size Composite is always equal. */
  friend bool operator==(const Composite&, const Composite&) = default;

 private:
  /** Calls \a f(block, rows) with each block and its rows of \a rows, an operand of values. */
  template <typename Derived, typename F>
  void forEachBlock(const MatrixBase<Derived>& rows, F f) const {
    auto& m = writable(rows);
    eigen_assert(m.rows() == valueDimension() && "the rows do not match the space");
    Index offset = 0;
    [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      ((f(block<Idx>(),
          internal::middleRowsOf<valueDimensionAtCompileTime<std::tuple_element_t<Idx, std::tuple<Blocks...>>>>(
              m, offset, block<Idx>().valueDimension())),
        offset += block<Idx>().valueDimension()),
       ...);
    }(std::make_index_sequence<BlockCount>{});
  }

  /** Calls \a f with each block and its rows of \a first, \a second, and \a out. An operand
   * whose flag is set holds differences, so a block takes its difference space's size there.
   */
  template <bool FirstDifference, bool SecondDifference, bool OutDifference, typename First, typename Second,
            typename Out, typename F>
  void forEachBlockOf(const MatrixBase<First>& first, const MatrixBase<Second>& second, const MatrixBase<Out>& out,
                      F f) const {
    auto& o = writable(out);
    Index firstOffset = 0;
    Index secondOffset = 0;
    Index outOffset = 0;
    const auto step = [&]<typename B>(const B& b) {
      const Index firstRows = internal::rowsOf<FirstDifference>(b);
      const Index secondRows = internal::rowsOf<SecondDifference>(b);
      const Index outRows = internal::rowsOf<OutDifference>(b);
      f(b, internal::middleRowsOf<internal::rowsAtCompileTime<FirstDifference, B>>(first, firstOffset, firstRows),
        internal::middleRowsOf<internal::rowsAtCompileTime<SecondDifference, B>>(second, secondOffset, secondRows),
        internal::middleRowsOf<internal::rowsAtCompileTime<OutDifference, B>>(o, outOffset, outRows));
      firstOffset += firstRows;
      secondOffset += secondRows;
      outOffset += outRows;
    };
    [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      (step(block<Idx>()), ...);
    }(std::make_index_sequence<BlockCount>{});
    eigen_assert(firstOffset == first.rows() && secondOffset == second.rows() && outOffset == o.rows() &&
                 "the rows do not match the space");
  }
};

/** \ingroup Spaces_Module
 * \brief A fixed-size \a Block repeated count() times, up to \a MaxCount_, such as one
 * block per vehicle.
 *
 * Its size is count() times the block's size. Every repetition shares the block's
 * parameters (periods and bounds). A tag derived from it can inherit its constructors to
 * start with a count: <tt>struct Vehicles : Repeated<Vehicle> { using Repeated::Repeated; };</tt>,
 * then <tt>Vehicles{Vehicle{}, 3}</tt>.
 */
template <Space Block, int MaxCount_ = Dynamic>
class Repeated {
  static_assert(Block::ValueDimensionStep == 0 && Block::FixedValueDimension > 0,
                "a repeated block must have a fixed, nonzero size");
  static_assert(MaxCount_ == Dynamic || MaxCount_ >= 0);

 public:
  /** The repeated block's type. */
  using BlockType = Block;

  /** Maximum number of repetitions, or \c Dynamic if unbounded. */
  static constexpr int MaxCount = MaxCount_;

  static constexpr int FixedValueDimension = 0;  ///< nothing is fixed: every coordinate belongs to a repetition
  static constexpr int ValueDimensionStep = Block::FixedValueDimension;  ///< the coordinates of one repetition

  /** \c MaxCount repetitions' worth of coordinates, or \c Dynamic if unbounded. */
  static constexpr int MaxValueDimension = MaxCount == Dynamic ? Dynamic : MaxCount * Block::FixedValueDimension;

  Repeated() = default;

  /** \a count repetitions of \a block. */
  explicit Repeated(Block block, Index count = 0) : m_block{std::move(block)} { setCount(count); }

  /** \returns the repeated block, with its parameters */
  const Block& block() const { return m_block; }

  /** \returns the number of repetitions */
  Index count() const { return m_count; }

  /** \returns the number of coordinates: count() repetitions of \c ValueDimensionStep */
  Index valueDimension() const { return m_count * ValueDimensionStep; }

  /** Sets the number of repetitions to \a count, at most \c MaxCount. */
  void setCount(Index count) {
    eigen_assert(count >= 0 && (MaxCount == Dynamic || count <= MaxCount) && "the dimension does not fit the space");
    m_count = count;
  }

  /** Resizes to \a n coordinates, a whole number of repetitions. */
  void resize(Index n) {
    eigen_assert(n >= 0 && n % ValueDimensionStep == 0 && "the dimension does not fit the space");
    setCount(n / ValueDimensionStep);
  }

  /** Normalizes each repetition; a repeated block without normalize() has none. */
  template <typename Derived>
  void normalize(const MatrixBase<Derived>& rows) const
    requires(internal::HasNormalize<Block>)
  {
    forEachRepetition(rows, [this](auto&& sub) { Eigen::normalizeValues(m_block, sub); });
  }

  /** \f$ x \boxplus \delta \f$ for each repetition, for a block with its own plus(). */
  template <typename X, typename D, typename Out>
  void plus(const MatrixBase<X>& x, const MatrixBase<D>& d, const MatrixBase<Out>& out) const
    requires(internal::HasPlus<Block>)
  {
    forEachRepetitionOf<false, true, false>(
        x, d, out, [this](const auto& xs, const auto& ds, const auto& os) { internal::plusInto(m_block, xs, ds, os); });
  }

  /** \f$ a \boxminus b \f$ for each repetition, for a block with its own minus(). */
  template <typename A, typename B, typename Out>
  void minus(const MatrixBase<A>& a, const MatrixBase<B>& b, const MatrixBase<Out>& out) const
    requires(internal::HasMinus<Block>)
  {
    forEachRepetitionOf<false, false, true>(a, b, out, [this](const auto& as, const auto& bs, const auto& os) {
      internal::minusInto(m_block, as, bs, os);
    });
  }

  /** The block's difference space, repeated as many times, keeping its tag as Composite's does. A repeated linear block
   * is its own.
   */
  auto difference() const
    requires(!isLinear<Block>)
  {
    using Difference = typename internal::DifferenceBlockOf<Block>::type;
    return Repeated<Difference, MaxCount_>{internal::DifferenceBlockOf<Block>::make(m_block), m_count};
  }

  /** Equal when every stored parameter and size is; a fixed-size Repeated is always equal. */
  friend bool operator==(const Repeated&, const Repeated&) = default;

 private:
  /** Calls \a f(rows) with each repetition's rows of \a rows, an operand of values. */
  template <typename Derived, typename F>
  void forEachRepetition(const MatrixBase<Derived>& rows, F f) const {
    auto& m = writable(rows);
    eigen_assert(m.rows() == valueDimension() && "the rows do not match the space");
    for (Index k = 0; k < m_count; ++k) {
      f(internal::middleRowsOf<ValueDimensionStep>(m, k * ValueDimensionStep, ValueDimensionStep));
    }
  }

  /** Calls \a f with each repetition's rows of \a first, \a second, and \a out; see
   * Composite::forEachBlockOf().
   */
  template <bool FirstDifference, bool SecondDifference, bool OutDifference, typename First, typename Second,
            typename Out, typename F>
  void forEachRepetitionOf(const MatrixBase<First>& first, const MatrixBase<Second>& second, const MatrixBase<Out>& out,
                           F f) const {
    auto& o = writable(out);
    const Index firstRows = internal::rowsOf<FirstDifference>(m_block);
    const Index secondRows = internal::rowsOf<SecondDifference>(m_block);
    const Index outRows = internal::rowsOf<OutDifference>(m_block);
    eigen_assert(first.rows() == m_count * firstRows && second.rows() == m_count * secondRows &&
                 o.rows() == m_count * outRows && "the rows do not match the space");
    for (Index k = 0; k < m_count; ++k) {
      f(internal::middleRowsOf<internal::rowsAtCompileTime<FirstDifference, Block>>(first, k * firstRows, firstRows),
        internal::middleRowsOf<internal::rowsAtCompileTime<SecondDifference, Block>>(second, k * secondRows,
                                                                                     secondRows),
        internal::middleRowsOf<internal::rowsAtCompileTime<OutDifference, Block>>(o, k * outRows, outRows));
    }
  }

  EIGEN_NO_UNIQUE_ADDRESS Block m_block;  ///< the repeated block, with its parameters
  Index m_count = 0;                      ///< the number of repetitions
};

namespace internal {

/** \internal Declared, never defined: matched by a pointer to a Composite or a tag derived from one, so
 * that IsComposite and CompositeBlocks see through the derivation.
 */
template <typename... Blocks>
std::tuple<Blocks...>* compositeBlocks(const Composite<Blocks...>*);

/** \internal A Composite, or a tag derived from one. */
template <typename T>
concept IsComposite = requires(const T* t) { compositeBlocks(t); };

/** \internal The blocks of the Composite \a T, as a tuple type. */
template <typename T>
using CompositeBlocks = std::remove_pointer_t<decltype(compositeBlocks(std::declval<const T*>()))>;

/** \internal resizableBlockCount summed over the blocks in the tuple type \a Blocks. */
template <typename Blocks>
inline constexpr int resizableBlockCountIn = 0;

template <typename... Blocks>
inline constexpr int resizableBlockCountIn<std::tuple<Blocks...>> = (0 + ... + resizableBlockCount<Blocks>);

/** \internal The resizable blocks of a Composite, or of a tag derived from one: its blocks' together. */
template <IsComposite S>
inline constexpr int resizableBlockCount<S> = resizableBlockCountIn<CompositeBlocks<S>>;

/** \internal A space that takes its sizes from a length alone: it has at most one resizable block, at any
 * depth. A space with several takes its counts from the space a caller gives, so
 * that a wrong split of the right total length (2 tracks and 5 landmarks against data for 3 and
 * 3) can't go unnoticed.
 */
template <typename S>
concept SizedByLength = resizableBlockCount<S> <= 1;

/** \internal Declared, never defined: matched by a pointer to a Repeated or a tag derived from one. */
template <typename Block, int MaxCount>
Block* repeatedBlock(const Repeated<Block, MaxCount>*);

/** \internal A Repeated, or a tag derived from one. */
template <typename T>
concept IsRepeated = requires(const T* t) { repeatedBlock(t); };

/** \internal The repeated block of the Repeated \a T. */
template <typename T>
using RepeatedBlock = std::remove_pointer_t<decltype(repeatedBlock(std::declval<const T*>()))>;

/** \internal Declared, never defined: true for a pointer to a LeafExtent or a type derived from one. */
template <int Dimension, int MaxDimension>
std::true_type leafExtent(const LeafExtent<Dimension, MaxDimension>*);

std::false_type leafExtent(const void*);

/** \internal A Linear or Periodic space (or a tag derived from one): every coordinate alike. */
template <typename T>
concept IsUniformLeaf = decltype(leafExtent(std::declval<const T*>()))::value;

/** \internal Declared, never defined: true for a pointer to a Periodic or a type derived from one. */
template <auto U, PeriodicInterval Interval, int Dimension, int MaxDimension>
std::true_type periodicLeaf(const QPeriodic<U, Interval, Dimension, MaxDimension>*);

std::false_type periodicLeaf(const void*);

/** \internal A Periodic space, or a tag derived from one. */
template <typename T>
concept IsPeriodic = decltype(periodicLeaf(std::declval<const T*>()))::value;

/** \internal Where the values of a leaf block lie: <tt>[lower, upper)</tt> wrapping with period
 * <tt>upper - lower</tt> if \c periodic, else the whole line.
 */
struct LeafSupport {
  double lower = -std::numeric_limits<double>::infinity();  ///< the lower end
  double upper = std::numeric_limits<double>::infinity();   ///< the upper end
  bool periodic = false;                                    ///< whether values wrap at the ends
};

/** \internal \returns the support of the leaf block \a Leaf: the whole line for a linear one, its period
 * for a Periodic one, and nothing for a non-linear space written from scratch, whose range the
 * library can't see.
 */
// constexpr rather than consteval: MSVC 14.50 can't apply -> to a consteval call's optional result.
template <typename Leaf>
constexpr std::optional<LeafSupport> leafSupport() {
  if constexpr (IsPeriodic<Leaf>) {
    return LeafSupport{.lower = Leaf::lower(), .upper = Leaf::upper(), .periodic = true};
  } else if constexpr (isLinear<Leaf>) {
    return LeafSupport{};
  } else {
    return std::nullopt;
  }
}

/** \internal Calls \a f(offset, leaf) for each leaf block of \a space, in order, where \a offset is the
 * leaf's first coordinate: each Linear or Periodic block, each repetition of a
 * Repeated block's leaves, and a space written from scratch as a whole.
 */
template <typename S, typename F>
void forEachLeaf(const S& space, F&& f, Index offset = 0) {
  if constexpr (IsComposite<S>) {
    [&]<std::size_t... Idx>(std::index_sequence<Idx...>) {
      ((forEachLeaf(space.template block<Idx>(), f, offset), offset += space.template block<Idx>().valueDimension()),
       ...);
    }(std::make_index_sequence<std::tuple_size_v<CompositeBlocks<S>>>{});
  } else if constexpr (IsRepeated<S>) {
    for (Index k = 0; k < space.count(); ++k) {
      forEachLeaf(space.block(), f, offset + k * S::ValueDimensionStep);
    }
  } else {
    f(offset, space);
  }
}

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_SPACES_COMPOSITE_H
