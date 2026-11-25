# Decisions

The settled decisions behind this fork of Eigen, which builds the Space concept of EmrysFilter into
Eigen's own types so that a vector or matrix carries a unit per block: for each, the rule, the
reason, the alternatives worth not proposing again, and where the code is. It is written for
contributors. The order of the work is in [roadmap.md](roadmap.md), and the decisions behind the spaces
themselves, brought over from EmrysFilter with the code, are in [spaces-decisions.md](spaces-decisions.md). The fork
keeps Eigen's own guides for everything these files do not override: `AGENTS.md` and `.agents/`.

How it is kept: as EmrysFilter's `docs/design/decisions.md`. A decision is named in kebab case and
cited in brackets, `[spaces-are-shape]`, in headers, tests, and documents; it is added in the
commit that implements it, rewritten in place when it changes, and removed when it no longer
records a choice. **Rule**, **Why**, **Rejected**, and **Where**; mechanism belongs in the header.

## Index

| Decision | One line |
|---|---|
| [cxx20-baseline](#cxx20-baseline) | Supported headers compile as C++20 |
| [spaces-are-shape](#spaces-are-shape) | A space replaces Rows, Cols, MaxRows, and MaxCols; the integers are derived |
| [q-names](#q-names) | `QMatrix`, `QArray`, `QVector`; `Matrix` and its family are aliases on `Linear` spaces |
| [space-first](#space-first) | Spaces first, then the scalar, which is the representation type and defaults to `double` |
| [mandatory-units](#mandatory-units) | Every leaf space has a unit; the plain spaces are dimensionless |
| [dual-spaces](#dual-spaces) | `Dual<S>` types the column side of a linear map; Hart's rules type every operation |
| [typed-blocks](#typed-blocks) | A tag addresses a block; a block by offset needs a uniform axis |
| [decomposition-boundaries](#decomposition-boundaries) | Kernels compute on the scalar buffer; decompositions type their factors at the API |
| [continuous-spaces-only](#continuous-spaces-only) | Discrete spaces stay in EmrysFilter |
| [autodiff-supported](#autodiff-supported) | AutoDiff is a supported module, and `Eigen/Spaces` builds on it |
| [public-space-helpers](#public-space-helpers) | What EmrysFilter needs from the spaces is public API |
| [bibliography-page](#bibliography-page) | The papers the headers cite resolve in Eigen's own documentation |

## cxx20-baseline

*Supported headers compile as C++20.*

**Rule.** Every header under `Eigen/` and `contrib/Eigen/` requires C++20. New code uses concepts,
`requires` clauses, structural non-type template parameters, `consteval`, `if constexpr`, and
designated initializers directly; `EIGEN_IF_CONSTEXPR` is kept only where it already is. C++23 is
not required: EmrysFilter needs it for `std::expected`, which nothing here uses.

**Why.** A space is a concept over types with structural parameters (an interval, a unit), and the
space algebra below is constant-expression code over them. Keeping Core at C++14 would have forced
the spaces into a wrapper outside Eigen's expression templates, which is the design this fork
exists to avoid.

**Rejected.** A C++14 Core with hooks and a C++20 contrib module holding the spaces: the typing
must reach Eigen's own expressions and decompositions, so it cannot live beside them. C++23, for
nothing it would buy here.

**Where.** `CMakeLists.txt` (`cxx_std_20` and `CMAKE_CXX_STANDARD`), `Eigen/src/Core/util/Macros.h`
(the version check), `AGENTS.md` rule 3, `.agents/conventions.md`, `scripts/check_style.py`, the
SYCL CMake modules, and the CI tidy scripts.

## spaces-are-shape

*A space replaces Rows, Cols, MaxRows, and MaxCols; the integers are derived.*

**Rule.** The plain object types take a row space and a column space where they took four
integers. A space's `FixedValueDimension`, `ValueDimensionStep`, and `MaxValueDimension` give
`RowsAtCompileTime` and `MaxRowsAtCompileTime` as `valueDimensionAtCompileTime<S>` and
`maxValueDimensionAtCompileTime<S>`, and the rest of Eigen reads those constants as it does today.
`Linear<N>` is today's `N`, `Linear<Dynamic>` is `Dynamic`, and `Linear<Dynamic, M>` is `Dynamic`
with a maximum of `M`. Traits carry `RowSpace` and `ColSpace`, and every expression that changes
shape (products, transposes, blocks, replication, reshaping, reductions) computes its spaces, with
the integers following. Spaces are types, never template arguments by value.

**Why.** The triple is already a generalization of Eigen's pair for one axis, so making it primary
adds no second source of truth, and the derived integers keep the 850 uses of the compile-time
sizes in `Eigen/src` unchanged. What moves is the code that builds plain types from integers: the
plain-type helpers in `XprHelper.h`, the plain objects, `Block`, `Map`, and `Ref`.

**Rejected.** A traits-level `RowSpace` beside the integers, both authoritative: two things to keep
consistent in every expression. A typed matrix wrapping an Eigen matrix (TypedLinearAlgebra's
pattern, and EmrysFilter's `GaussianParams`): it cannot type Eigen's expressions, blocks, or
decompositions, and it fixes the size at compile time. Spaces as `auto` template arguments by
value, to keep one `Matrix` name for both spellings: every tag and `Composite` would have to be
structural, and users would write `Position{}` for a type.

**Where.** `Eigen/src/Core/Matrix.h`, `Array.h`, `PlainObjectBase.h`, `DenseStorage.h`,
`util/XprHelper.h`, `util/ForwardDeclarations.h`, `Block.h`, `Map.h`, `Ref.h`.

## q-names

*`QMatrix`, `QArray`, `QVector`; `Matrix` and its family are aliases on `Linear` spaces.*

**Rule.** The space-typed plain objects are `QMatrix<RowSpace, ColSpace, Scalar, Options>` and
`QArray<RowSpace, ColSpace, Scalar, Options>`, with the aliases `QVector<Space, Scalar>` and
`QRowVector<Space, Scalar>` putting `Linear<1>` on the other axis. Q is for quantity, the name
mp-units gives a number with a unit. `Matrix`, `Array`, `Vector`, `RowVector`, and every typedef
built on them (`Matrix3d`, `VectorXf`, ...) become alias templates onto the Q types with `Linear`
spaces and the dimensionless unit, keeping their old signatures: `Matrix<double, 3, 1>` is
`QMatrix<Linear<3>, Linear<1>, double>`. Nothing written against the old names changes. The
deprecated conversions to Qt's `QMatrix` and `QTransform` in `Geometry/Transform.h` are removed, so
that the name has one meaning inside Eigen.

**Why.** The old spelling appears 603 times in `Eigen/src`, 1533 in `contrib`, and 2773 in `test`;
as an alias it costs nothing and keeps the test suite as the check on the refactor. A qualifying
prefix on `Matrix` is Eigen's convention (`SparseMatrix`, `DiagonalMatrix`), and Q is the shortest
one that says what the type is. The collision with Qt's class names is accepted: the names live in
`Eigen::`, and the only place the tree used Qt's was already deprecated.

**Rejected.** `QuantityMatrix`: right, and too long in every signature. `Dimensioned`, Hart's term:
"dimension" means a size in `valueDimension()` and in most readers' heads, and the two meanings
would sit side by side. `UnitMatrix`: a unit matrix is the identity, and `UnitX()` is a unit vector.
`Quantity`: mp-units' concept, which a `using namespace mp_units` brings into scope. Giving
`Matrix` the new signature and renaming the alias: it breaks every user and the tests at once, for
nothing.

**Where.** `Eigen/src/Core/util/ForwardDeclarations.h`, `Matrix.h`, `Array.h`,
`Eigen/src/Geometry/Transform.h`.

## space-first

*Spaces first, then the scalar, which is the representation type and defaults to `double`.*

**Rule.** The Q types take the row space, the column space, the scalar defaulting to `double`, and
`Options` last, whose default (row-major for a row vector) is computed from the spaces as it is
from the integers today. The scalar is the representation type, mp-units' `Rep`: the type stored,
with no unit of its own. Units live in the spaces.

**Why.** The spaces are the point of the type, and `QMatrix<State, State> P;` is how EmrysFilter
already names a distribution's space before its scalar. The aliases insulate Eigen's scalar-first
convention. A representation type is a property of the storage, not of either axis; a scalar on
each leaf would let the two axes disagree.

**Rejected.** Eigen's order, scalar first, for the Q types. A scalar carried by the leaf spaces.

**Where.** `Eigen/src/Core/util/ForwardDeclarations.h`.

## mandatory-units

*Every leaf space has a unit; the plain spaces are dimensionless.*

**Rule.** A leaf space carries one unit for all its coordinates, as its first template argument:
`QLinear<U, N, Max>` and `QPeriodic<U, Interval, N, Max>`, with the named angles `QRadiansPi<U, N,
Max>` and the other three. The plain names are the dimensionless aliases, `Linear<N, Max>` for
`QLinear<dimensionless, N, Max>` and likewise `Periodic`, `RadiansPi`, `Radians2Pi`, `Degrees180`,
and `Degrees360`, so every spelling written before the unit existed keeps its type, and a space
written from scratch declares no unit and is dimensionless. The unit is first so that the maximum
dimension need not be written to give it; the dimension is second; the maximum stays last. A unit is
any structural value satisfying Eigen's `Unit` concept: closed under product and quotient, with
`constexpr` equality, and not a number; `Eigen::dimensionless` is the identity, which a foreign unit
absorbs on either side of a product or quotient and equals exactly when it is the dimensionless unit
of its own system. Eigen defines the concept and `dimensionless` and nothing else; mp-units' unit
objects satisfy it as they are. `unitAt<S, Path>()` reads the unit of the block a path names, and
`unitOf<S>()` a space's where its blocks agree. `QPeriodic`'s interval is read in its unit, so
`QDegrees360<si::degree>` and `QRadians2Pi<si::radian>` differ in unit and not only in period.

**Why.** One unit per leaf, not a tuple per space: a leaf is already uniform by construction (one
normalize rule, one period), a mixed-unit state is already a Composite of tagged leaves, and a
resizable or repeated block takes one unit where a tuple the length of the space could not be
written. One rule in every check, product, and dual, instead of a unit case and a no-unit case. The
Q prefix follows [q-names]: no Q, no units, for leaves as for matrices.

**Rejected.** An optional unit. mp-units as a dependency of Core. A tuple of units the length of the
space. The unit as a trailing parameter, `Linear<3, 3, si::metre>`, which made the rarest parameter
mandatory; as the second, `Linear<3, si::metre>`, which broke `Linear<Dynamic, 8>`. Trailing letters
for the dimensionless alias on the model of `MatrixXd`, `LinearD`: those letters name concrete sizes
and scalars, and `D` already reads as dynamic or double. A `Quantity<Unit, Space>` adaptor.

**Where.** `Eigen/src/Core/Spaces/Unit.h`, `Coordinates.h`, `Dual.h`; `test/spaces_units.cpp`.

## dual-spaces

*`Dual<S>` types the column side of a linear map; Hart's rules type every operation.*

**Rule.** Entry (i, j) of `QMatrix<R, C>` has unit u_R(i) u_C(j). `Dual<S>` is a wrapper,
`DualSpace<S>`, that keeps the blocks and tags of `S` and reports every unit inverted, so that a tag
path into X names the same block of `Dual<X>`; it is linear, since covectors add as plain vectors.
The alias collapses: `Dual<Dual<S>>` is `S`, and a space whose every unit is dimensionless, tagged
or not, is its own dual, so a plain matrix's spaces meet the product rule with no special case. A
linear map from X to Z is `QMatrix<Z, Dual<X>>`, a covariance of x is `QMatrix<X, X>`, and an
information matrix is `QMatrix<Dual<X>, Dual<X>>`. The operations:

- sum, difference, and cwise operations: equal spaces, where `Linear<Dynamic>` is compatible with
  `Linear<n>` of the same unit and keeps today's run-time size check;
- product: `C_lhs` is `Dual<R_rhs>`, result `(R_lhs, C_rhs)`; the inner-dimension static assert is
  the dimensionless case;
- transpose swaps; inverse is `(Dual<C>, Dual<R>)`; a solve of `A x = b` puts x in `Dual<C>`;
- a reduction along an axis, and `colwise()` or `rowwise()`, require every coordinate of that axis
  to have one unit; `norm()` and `dot()` likewise on their vector.

These are the rules of Hart, *Multidimensional Analysis* (Springer, 1995), chapters 2 to 4.

**Why.** A Kalman filter types end to end under them: P Hᵀ S⁻¹ is `(X, Dual<Z>)`, K times the
innovation is in X, and P x does not compile. Eigenvalues of a mixed-unit covariance do not compile
either, which is the classic mistake they catch. Units per block, not per entry, is what lets a
`Dynamic` axis stay typed.

**Rejected.** A type per row and per column (TypedLinearAlgebra): N + M types, no dynamic sizes.
Typing through the scalar, `Matrix<quantity<U>>`, as mp-units and Au do: one unit for the whole
matrix, so no state vector and no covariance. `Dual<X>` as the leaf with its unit inverted, which
would drop the tags and with them addressing on the column side of a Jacobian.

**Where.** `Eigen/src/Core/Spaces/Dual.h`; the rules in `CwiseBinaryOp.h`, `Product.h`,
`Transpose.h` (step 3), and `Inverse.h`, `Solve.h`, `Redux.h`, `VectorwiseOp.h`, `Dot.h` (step 5).

## typed-blocks

*A tag addresses a block; a block by offset needs a uniform axis.*

**Rule.** A tag or tag path is an index argument: `x(Position{})`, `P(Position{}, Velocity{})`,
and the compile-time forms `m.rows<Path>()`, `m.cols<Path>()`, and `m.block<RowPath, ColPath>()`
resolve the path against the matrix's own row or column space, as EmrysFilter's `idx<Path>()` did
against a space the caller passed, and return blocks typed by the subspaces named. A covariance's
row space is a difference space, so the same path names the difference block on it and the value
block on a mean; `idx()` and `diffIdx()` have no successor. A block at run-time offsets,
`block(i, j, p, q)` and the `segment`, `head`, and `tail` family, is allowed on an axis whose
coordinates share one unit and returns a `Linear` of that unit, so every plain matrix behaves as
today; on an axis of mixed units it does not compile. `m.untyped()` is the explicit view with every
unit dropped, for code that must see the buffer.

**Why.** An offset chosen at run time can name coordinates of different units, so there is nothing
the result could be typed; a path can, and does. The matrix carries its spaces, so the three things
a caller of `idx()` had to supply, the space, whether the operand held values or differences, and
the block call, are all read off the type. Keeping offsets legal on uniform axes is what leaves
Eigen's own code and tests untouched.

**Rejected.** Untyping every run-time block silently. Forbidding run-time blocks on typed spaces
altogether, which would exclude `Repeated` blocks of one unit. Keeping `idx()` beside the typed
forms, which would keep the untyped index range in user code.

**Where.** `Eigen/src/Core/Block.h`, `DenseBase.h`, `IndexedView.h`, `Spaces/TagPath.h`.

## decomposition-boundaries

*Kernels compute on the scalar buffer; decompositions type their factors at the API.*

**Rule.** Product kernels, packet code, and the decomposition algorithms operate on the untyped
scalar buffer. Each decomposition states the spaces of its inputs and factors in its public
interface: the Cholesky factor of `QMatrix<X, X>` is `QMatrix<X, Linear<n>>`; an eigendecomposition
takes an endomorphism, `QMatrix<X, Dual<X>>`, so a covariance has eigenvalues only when X is
uniform; a determinant takes a fixed-size matrix, since its unit is the product of the diagonal's.
The remaining factorizations state their factor spaces as they are converted, following Hart.

**Why.** The numerics do not change, and nothing in the kernels needs to know a unit. The
boundary is where a factor acquires a meaning, and it is where a misuse is caught.

**Rejected.** Typing inside the kernels. Leaving decompositions untyped and casting at each use.

**Where.** `Eigen/src/Cholesky`, `LU`, `QR`, `SVD`, `Eigenvalues`; the public classes only.

## continuous-spaces-only

*Discrete spaces stay in EmrysFilter.*

**Rule.** The fork takes EmrysFilter's continuous spaces: `Linear`, `Periodic`, `Composite`,
`Repeated`, `UnitQuaternion`, the tag paths, and the reset. Its discrete spaces, `Finite`,
`Named`, and `DiscreteProduct`, stay in EmrysFilter with their consumers, `Categorical`, the Markov
chain, and the switching models. `QMatrix` takes two `Space`s, and `Space` is EmrysFilter's concept
plus a unit.

**Why.** A discrete space has the shape an axis needs and nothing an axis is typed for: its entries
are dimensionless probabilities with no ⊞ and no difference space, so every piece of the unit and
dual machinery is trivial on it. What a discrete axis would have bought, named indexing into a
probability vector and a check between two mode sets of equal size, does not pay for 748 lines in
Core and a concept generalized to hold a family that contributes nothing to it.

**Rejected.** An `Axis` concept below `Space` and `DiscreteSpace`, so that both type a `QMatrix`.

**Where.** EmrysFilter's `Spaces/Discrete.h`, which keeps its own copies of the part storage and
the tag-path resolution it shares with `Composite`.

## autodiff-supported

*AutoDiff is a supported module, and `Eigen/Spaces` builds on it.*

**Rule.** `contrib/Eigen/AutoDiff` moves to `Eigen/AutoDiff`, with the old path forwarding. The
reset of a space's covariance to the tangent space at a moved mean (EmrysFilter's `Reset.h`) lives
in `Eigen/Spaces`, an umbrella that includes Core, LU, and AutoDiff: the identity and closed-form
paths, and the fallback that differentiates a space's plus() and minus() with `AutoDiffScalar`.
The leaves, `Composite`, the tag paths, `Dual`, and the unit concept live under
`Eigen/src/Core/Spaces/`, where `Matrix` needs them.

**Why.** Nothing under `Eigen/` may include contrib, and the fallback is what makes a space written
from scratch work without a hand-written Jacobian. `AutoDiffScalar` is already what EmrysFilter
differentiates every model with, so it is promoted on merit.

**Rejected.** The fallback in contrib, separate from the reset it completes. Core depending on
LU.

**Where.** `Eigen/AutoDiff`, `Eigen/Spaces`, `Eigen/src/Spaces/Reset.h`.

## public-space-helpers

*What EmrysFilter needs from the spaces is public API.*

**Rule.** A helper of the Spaces layer that EmrysFilter calls from outside that layer becomes a
documented public name in `Eigen::`, named for what it is, when it comes over. The tag-addressing
forms replace `idx()`. No downstream code reaches into `Eigen::internal`.

**Why.** EmrysFilter reaches into its own `Internal` for the index resolution in 9 files, its
single-row-aware plain matrix alias in 15, and its symmetric-block, selection, and identity-reset
helpers in 7 each. Left internal, the port would turn its author's own library into a user of
Eigen's private namespace.

**Rejected.** EmrysFilter using `Eigen::internal` by special dispensation.

**Where.** `Eigen/src/Core/Spaces/`, decided helper by helper as the port proceeds.

## bibliography-page

*The papers the headers cite resolve in Eigen's own documentation.*

**Rule.** `doc/Bibliography.dox` holds one anchor per cited work, keyed as EmrysFilter's
bibliography keys them (`Hertzberg2013`, `Sola2018`, ...), so a `\ref` in a header resolves and
EmrysFilter's citations keep resolving when it points at the fork's documentation.

**Why.** Eigen's documentation job fails on one unresolved reference, and the spaces cite six
works; the alternative, prose citations in every header, loses the link.

**Rejected.** Rewriting the citations as plain text. Doxygen's `\cite` with a BibTeX file, which
needs a tool the documentation image does not have.

**Where.** `doc/Bibliography.dox`.
