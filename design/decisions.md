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
| [mp-units-dependency](#mp-units-dependency) | The typed spaces are built on mp-units; plain Eigen does not include it |
| [leaf-quantities](#leaf-quantities) | A space fixes what its coordinates measure; a unit is bound to it, and a tag names it without one |
| [dual-spaces](#dual-spaces) | `Dual<S>` types the column side of a linear map; Hart's rules type every operation |
| [heterogeneous-results](#heterogeneous-results) | Compatible operands combine as mp-units quantities do |
| [additive-groups](#additive-groups) | Coordinate arithmetic is a space's own only in a group under addition |
| [complex-scalars](#complex-scalars) | Units do not depend on the scalar being complex |
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
with a maximum of `M`. An expression's spaces are `internal::spaces_of<Xpr>::RowSpace` and
`ColSpace`, and every expression that changes shape (products, transposes, blocks, replication,
reshaping, reductions) specializes it to compute them, with the integers following. An expression
without a specialization has dimensionless `Linear` spaces of its sizes. The plain objects name
their spaces as the member types `RowSpace` and `ColSpace`. Spaces are types, never template
arguments by value.

**Why.** The triple is already a generalization of Eigen's pair for one axis, so making it primary
adds no second source of truth, and the derived integers keep the 850 uses of the compile-time
sizes in `Eigen/src` unchanged. What moves is the code that builds plain types from integers: the
plain-type helpers in `XprHelper.h`, the plain objects, `Block`, `Map`, and `Ref`. The spaces are
kept out of `traits` because `Transpose`, `Block`, `Map`, and others derive their traits from their
argument's: a `RowSpace` in `traits<QMatrix>` would pass to a transpose unchanged, which for a
square matrix is the wrong unit with no error. A trait each expression must opt into fails safe
instead, to plain sizes.

**Rejected.** A traits-level `RowSpace` beside the integers, both authoritative: two things to keep
consistent in every expression. `RowSpace` in `traits` with the integers derived from it: inherited
silently by every expression whose traits derive from its argument's. A typed matrix wrapping an Eigen matrix (TypedLinearAlgebra's
pattern, and EmrysFilter's `GaussianParams`): it cannot type Eigen's expressions, blocks, or
decompositions, and it fixes the size at compile time. Spaces as `auto` template arguments by
value, to keep one `Matrix` name for both spellings: every tag and `Composite` would have to be
structural, and users would write `Position{}` for a type.

**Where.** `Eigen/src/Core/Matrix.h`, `Array.h`, `DenseBase.h` (`PlainObject`),
`util/XprHelper.h` (`spaces_of` and the plain-type helpers), `util/ForwardDeclarations.h`,
`Spaces/Linear.h`; `Block.h`, `Map.h`, `Ref.h` (step 5); `test/qmatrix.cpp`.

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

## mp-units-dependency

*The typed spaces are built on mp-units; plain Eigen does not include it.*

**Rule.** Quantity specs, units, references, their conversion factors, and their common units are
mp-units' own, used as they are; Eigen defines no unit system and no unit concept. `Eigen/Units`
includes mp-units and makes spaces with units usable. `Eigen/Core` keeps `dimensionless` and the
plain spaces and compiles without mp-units; it reaches mp-units only through
`internal::units_backend`, which Core declares and `Eigen/Units` defines, so a space with units used
without `Eigen/Units` fails to compile instead of meaning something else in that translation unit.
This reverses the decision of 2026-11-04 (2026-11-25).

**Why.** mp-units is proposed for the standard library (P3045), and the best outcome of this fork
is to show what Eigen 6 looks like on it. Conversion needs it: a common unit and a conversion
factor are read from mp-units' magnitudes, which a concept over `*`, `/`, and `==` cannot reach, so
metres and feet could only be rejected. Using its quantity specs gives the spaces kinds
(`isq::height` against `isq::width`) and the rules for which converts into which, instead of a
second design of them.

**Rejected.** Eigen's own `Unit` concept satisfied by any library (the decision reversed): no
conversion, and a second vocabulary beside the one the standard may adopt. `Eigen/Core` including
mp-units: every plain translation unit would pay for it. A customization point open to other unit
libraries: nothing needs it, and it would freeze an interface before the standard settles.

**Where.** `Eigen/Units`, `Eigen/src/Units/`; `Eigen/src/Core/Spaces/Unit.h` for `dimensionless`
and the declaration of `units_backend`; `cmake/EigenMPUnits.cmake`.

## leaf-quantities

*A space fixes what its coordinates measure; a unit is bound to it, and a tag names it without one.*

**Rule.** A leaf space's first template argument is what its coordinates measure: an mp-units
quantity spec, `QLinear<isq::displacement, 3>`, which fixes the dimension and the kind; a unit,
`QLinear<si::metre, 3>`, whose kind is the unit's as in `quantity<si::metre>`; or a reference,
`QLinear<isq::height[si::metre], 3>`. The plain names are the dimensionless aliases, `Linear<N,
Max>` for `QLinear<dimensionless, N, Max>` and likewise `Periodic` and the named angles, so every
spelling written before units keeps its type. A leaf with a unit is bound, a space whose every leaf
is bound is bound, and a `QMatrix`'s spaces are bound. A tag names a space without units, `struct
Position : QLinear<isq::displacement, 3> {};`, and a unit is bound to it apart, by In: by dimension,
`In<Position, si::metre>` and `In<State, si::metre, si::metre / si::second>`, each unit binding the
leaves without one whose quantity it measures and a leaf that two would fit not compiling; or by tag
at any depth, `In<State, bind<Position>(si::metre), bind<Velocity>(si::foot / si::second)>`. A
leaf with a unit keeps it, so a periodic leaf, whose interval is read in its unit, carries one from
its definition. A composite of bound blocks, `Composite<In<Position, si::metre>, ...>`, is bound
too, and a tag path names a bound block by its tag. The binding is kept by leaf position, so it
survives `difference()` and `Dual`, which keep the order of the leaves and not always their tags.
`QPeriodic`'s interval is read in its unit, so `QDegrees360<si::degree>` and
`QRadians2Pi<si::radian>` differ in unit and not only in period. `unitAt<S, Path>()` and
`unitOf<S>()` read units.

**Why.** A tag is to a space what `isq::height` is to `isq::height[m]`: `x(Position{})` names the
position block in metres or in feet, converting a state keeps its tags, and a model written for
`Position` and `Velocity` works in any unit. One unit per leaf, not a tuple per space: a leaf is
uniform by construction, a mixed-unit state is a Composite of tagged leaves, and a resizable or
repeated block takes one unit where a tuple the length of the space could not be written.

**Rejected.** Tags that fix a unit, `struct Position : QLinear<si::metre, 3> {};`, the form of
2026-11-04: converting a state would change its tags. A unit without a quantity spec as the only
form: no kinds. Binding by position as the spelling, which goes wrong without failing when the
blocks are reordered; the same spelling binds by dimension instead. `In<State, In<Position,
si::metre>, ...>`, the first spelling of a binding by tag: a unit is a value and a tag a type, which
one template cannot take in the same place. Tags as values, so that
`position[si::metre]` reads as in mp-units: a space with run-time sizes is not a structural type.
An optional unit, a tuple of units the length of the space, the unit as a trailing parameter, and a
`Quantity<Unit, Space>` adaptor, as before.

**Where.** `Eigen/src/Core/Spaces/Unit.h`, `Linear.h`, `Coordinates.h`, `Bound.h`, `Dual.h`;
`TagPath.h` for `TagOf`; `test/spaces_units.cpp`, `test/units.cpp`; `failtest/space_bind_*.cpp`,
`failtest/space_matrix_unbound.cpp`.

## dual-spaces

*`Dual<S>` types the column side of a linear map; Hart's rules type every operation.*

**Rule.** Entry (i, j) of `QMatrix<R, C>` has unit u_R(i) u_C(j). `Dual<S>` is a wrapper,
`DualSpace<S>`, that keeps the blocks and tags of `S` and reports every unit inverted, so that a tag
path into X names the same block of `Dual<X>`; it is linear, since covectors add as plain vectors.
The alias collapses: `Dual<Dual<S>>` is `S`, and a space whose every unit is dimensionless, tagged
or not, is its own dual, so a plain matrix's spaces meet the product rule with no special case. A
linear map from X to Z is `QMatrix<Z, Dual<X>>`, a covariance of x is `QMatrix<X, X>`, and an
information matrix is `QMatrix<Dual<X>, Dual<X>>`. The operations:

- sum, difference, and cwise operations: spaces of one structure whose coordinates measure
  compatible quantities [heterogeneous-results], on a group under addition [additive-groups]. A
  cwise product or quotient is typed only against a dimensionless operand, as a scalar is;
- assignment, construction, `+=`, and `-=`: the source converts into the destination's spaces as
  mp-units converts a quantity [heterogeneous-results]; `*=` and `/=` take a dimensionless factor.
  `Zero()`, `Constant()`, and `Identity()` are in the spaces of the type they are called on;
- product: `C_lhs` is `Dual<R_rhs>` up to one common factor c, result `(R_lhs c, C_rhs)`
  [heterogeneous-results]; the inner-dimension static assert is the dimensionless case;
- transpose and adjoint swap; inverse is `(Dual<C>, Dual<R>)`; a solve of `A x = b` puts x in `Dual<C>`;
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

An expression the rules do not reach yet (a block, a map, a unary operation) has dimensionless
spaces, so it combines with untyped operands only: a false rejection, never a wrong unit, except that
it drops the units of a typed operand it wraps.

**Where.** `Eigen/src/Core/Spaces/Dual.h`; the rules in `Spaces/Rules.h`, checked in
`CwiseBinaryOp.h`, `Product.h`, and `AssignEvaluator.h`, with the spaces of `Transpose`, `Product`,
`CwiseBinaryOp`, and `CwiseNullaryOp` (step 3); `failtest/space_rule_*.cpp`; and `Inverse.h`,
`Solve.h`, `Redux.h`, `VectorwiseOp.h`, `Dot.h`, and the blocks and maps (step 5).

## heterogeneous-results

*Compatible operands combine as mp-units quantities do: common units, common kinds, tags kept only when shared.*

**Rule.** Two spaces are compatible where they have one structure and, coordinate by coordinate,
quantities that mp-units would add: quantity specs with a common spec, so units of one dimension. A
sum or difference is in the common space: at each coordinate the common unit (`get_common_unit`)
and the common spec (`get_common_quantity_spec`), with each operand converted into it, which does
not depend on the order of the operands. Its tags are the operands' where both have the same tags,
rebound to the common units, and otherwise the untagged leaves of the common quantities. Assignment
and construction convert the source into the destination's units where mp-units converts a
quantity implicitly (an implicitly convertible spec, and a floating-point `Scalar` or an integral
factor); otherwise `x.in<Space>()`, or `m.in<RowSpace, ColSpace>()`, converts explicitly. A
product's inner coordinates, `u_C_lhs(k) u_R_rhs(k)`, must share one dimension; where they differ in
unit the operands are converted to their common unit c, and the result is `(R_lhs c, C_rhs)`: each
unit of `R_lhs` times c. A c with a dimension changes the leaves' specs, which drops their tags, so
a plain `Matrix3d` times a position in metres is `QLinear<si::metre, 3>`; a c of a unit ratio, such
as m/ft, keeps them. Where the units already agree no conversion is generated.

**Why.** It is what mp-units does with `1 * m + 1 * ft` and `quantity<si::metre> q = 3 * ft`, so
the two read alike; an order-independent result is what makes `a + b` and `b + a` one type. A tag
held by only one operand would be a claim the other did not make.

**Rejected.** The left operand's spaces, Eigen's rule for sizes: `a + b` and `b + a` of different
types. Rejecting different units: mp-units adds them. Converting into the left operand's unit. The
nearest common base of two tags, which mp-units' hierarchy of specs would suggest: C++20 cannot list
a type's bases, so it would hold only for tags that name a parent; recorded for later.

**Where.** `Eigen/src/Core/Spaces/Rules.h`, `Eigen/src/Units/Rules.h`, `Conversion.h`;
`failtest/space_rule_*.cpp`, `failtest/units_*.cpp`; `test/units.cpp`.

## additive-groups

*Coordinate arithmetic is a space's own only in a group under addition.*

**Rule.** `+`, `-`, and the coefficient-wise operations act on coordinates, which is the space's
group product only where that product is addition: `Linear`, `Periodic` (exactly, modulo its
period, and unwrapped until `normalize()`), and a `Composite` or `Repeated` of such, as
EmrysFilter's `groups-prototype` classifies them. On any other space, `UnitQuaternion` or a space
with its own `plus()`, they do not compile, and `compose()`, `plus()`, and `minus()` are the
operations; a product's inner spaces are held to the same rule. `LieGroup`, `compose()`,
`inverse()`, `identity()`, `exp()`, and `log()` come from that branch with the classification.

**Why.** A difference of two headings by `-` is right modulo the period; of two quaternions it is
not a rotation. Requiring a linear space instead would reject a covariance over a state with a
heading, whose difference space is a `Periodic` on `[-p/2, p/2)`.

**Rejected.** Linear spaces only, for that reason. Treating every matrix as a tensor, so that the
check never applies: a vector of quaternions would add. Telling values from differences in the
type, as mp-units' `quantity_point` does: deferred.

**Where.** `Eigen/src/Core/Spaces/Group.h`, `Rules.h`; `Eigen/src/Geometry/UnitQuaternion.h`;
`failtest/space_rule_*.cpp`.

## complex-scalars

*Units do not depend on the scalar being complex.*

**Rule.** Entry (i, j) has the unit u_R(i) u_C(j) whether `Scalar` is real or complex, and
conversion factors are real. A leaf whose first argument names a quantity spec holds a `Scalar` of
the spec's character, as mp-units' `RepresentationOf` checks; a leaf given only a unit takes any, as
mp-units' `kind_of` references do. `adjoint()` swaps the spaces as `transpose()` does; `real()`,
`imag()`, and `abs()` keep units; `norm()` is |x|^2 in Eigen and squares them. A spec of vector
character on a leaf of N coordinates means the leaf holds that vector's components; recorded, not
checked.

**Why.** mp-units makes complex a property of the representation and the spec, never of the unit;
the same split keeps one rule for real and complex matrices.

**Rejected.** A complex unit, or a separate rule for complex matrices.

**Where.** `Eigen/src/Units/Rules.h`; `test/units.cpp`.

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
The leaves, `Composite`, the tag paths, `Dual`, `In`, and `dimensionless` live under
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
