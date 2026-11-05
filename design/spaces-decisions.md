# Decisions: the spaces

The decisions behind the Space layer, brought over from EmrysFilter's `docs/design/decisions.md` with the spaces
themselves (roadmap step 1) and adapted to where the code now lives. They are kept and cited as the decisions in
[decisions.md](decisions.md) are; the two files differ only in origin. `Spaces/` below is `Eigen/src/Core/Spaces/`
unless another path is given.

## Index

| Decision | One line |
|---|---|
| [sizes-and-dimensions](#sizes-and-dimensions) | A space has a value dimension and a difference dimension, and no bare `dimension()` |
| [index-names](#index-names) | The index utilities have short names: `idx()` and `diffIdx()` |
| [covariance-reset](#covariance-reset) | A covariance moves with its mean, by each block's reset Jacobian, and costs nothing where that Jacobian is the identity |
| [unit-quaternion](#unit-quaternion) | A unit quaternion space, stored as Eigen stores a quaternion and viewed through a map |
| [lie-groups](#lie-groups) | Lie groups are spaces; navigation is domain models written on them, elsewhere |
| [linear-space](#linear-space) | The plain space is named Linear, for its structure rather than its set |
| [no-bounded-coordinates](#no-bounded-coordinates) | There is no bounded coordinate type |
| [several-resizable-blocks](#several-resizable-blocks) | A composite may have several resizable blocks; only one with at most one takes its sizes from a length |
| [coordinate-parameters](#coordinate-parameters) | A coordinate type's parameters are template arguments, and an interval is spelled by its ends |
| [difference-spaces](#difference-spaces) | A space has at most one normalize(); its differences live in a difference space; updates and residuals are plus() and minus() |
| [tag-addressing](#tag-addressing) | Blocks are addressed by tag path, never by type or by index |
| [tangent-gaussians](#tangent-gaussians) | A Gaussian on a non-linear space is a Gaussian in the tangent space at its mean |
| [space-storage](#space-storage) | A space costs what its state costs |

## sizes-and-dimensions

*A space has a value dimension and a difference dimension, and no bare `dimension()`.*

**Rule.** `valueDimension()` is the length of a value, a mean or a sample, and `differenceDimension()` the rows of a
covariance and the degrees of freedom; there is no bare `dimension()`. A space implements only the member
`valueDimension()`, and generic code calls the free functions. `size()` is left to count elements, as it does in Eigen
and the standard library.

**Why.** A value and a difference may differ in size (a unit quaternion has 4 coordinates and 3 degrees of freedom), so
a distribution over a space has two dimensions, and one `size()` can't be both. Neither gets the bare name: the
manifold's dimension is the difference dimension, while most readers would take `dimension()` to be the length of the
mean.

**Rejected.** A space's `size()`, `FixedSize`, `Stride`, and `MaxSize`: a space's `size()` counted coordinates, against
the rule that `size()` counts elements. `Stride` (not a dimension) and `ElementValueDimension`, for what is now
`ValueDimensionStep`. `isValidValueDimension`, for what is now `allowsValueDimension`. A `constexpr` free
`valueDimension()`: it can't be constant for a resizable space or for a user's space without `constexpr` members, so
`valueDimensionAtCompileTime<S>` is the compile-time spelling, as Eigen has `RowsAtCompileTime` beside `rows()`.

**Where.** `Spaces/Space.h`.

## index-names

*The index utilities have short names.*

**Rule.** The functions that turn a tag path into an Eigen index are `idx()` for values and `diffIdx()` for
differences, an exception to spelled-out names. Each is one name with four overloads, told apart by their arguments:
`idx<Space, Path>()`, `idx<Path>(space)`, `idx<Path>(space, k)`, and `idx<Space, Path>(k)`. The first is a constant
expression where the path names a single coordinate. Both are replaced by tag addressing on the matrix itself when
the shape parameters become spaces ([typed-blocks](decisions.md#typed-blocks)).

**Why.** These names sit inside subscripts, often two to a call (`cov(diffIdx<S, RowPath>(), diffIdx<S, ColPath>())`),
where longer names pushed ordinary lines past the column limit and buried the paths, which are what the reader needs to
see. `diff` reads as "difference" without a lookup; a bare `d` could mean delta, derivative, or dimension.

**Rejected.** `differenceIdx`, for the length. A variable template for the compile-time form beside `idxIn()` and
`diffIdxIn()`: every form names its space, as a type or an object, so "In" said nothing the arguments don't.

**Where.** `Spaces/TagPath.h`.

## covariance-reset

*A covariance moves with its mean, by each block's reset Jacobian, and costs nothing where that Jacobian is the
identity.*

**Rule.** An update moves the mean by a step, $x^+ = x \boxplus \hat\delta$, and computes the posterior covariance in
the tangent space at $x$; it then moves the covariance to the tangent space at $x^+$ by the reset Jacobian $G$,
$P^+ \leftarrow G P G^T$. $G$ is a property of $\boxplus$, so the space supplies it, block by block: the identity for a
block without a `plus()` of its own; otherwise by forward-mode autodiff of the block's `plus()` and `minus()` by
default, by a closed form where the block has `resetJacobian()`, or as the identity where it declares
`identityReset`. Where every block has the identity, the reset compiles to nothing. The reset applies wherever a mean
moves by a step and keeps a covariance computed at the old mean: every Kalman-family update, and a Gaussian's
conditional. The information forms move their reference by the same $G$ (`internal::resetInfoAt()`).

**Why.** Without the reset a covariance is labeled with a mean it was not computed at, and a hand-written error-state
filter does the reset (Solà 2017, §6.3; Mueller, Hehn, and D'Andrea 2017; the closed form of Solà, Deray, and
Atchuthan 2018), so a library that skipped it would be wrong where hand code is right. Markley (2003) leaves the
covariance unchanged at the reset, which is the omission Mueller, Hehn, and D'Andrea correct. The Jacobian belongs to
the space because it is geometry: any update that moves a mean needs the same $G$. It is decided at compile time
because most spaces need nothing. Autodiff is the default because it is correct with no work from the space's author.

**Rejected.** A compile error for a space with its own `plus()` and no Jacobian: autodiff already makes the default
correct. A dense $n \times n$ $G$, which would multiply mostly by the identity.

**Evidence.** `test/spaces_reset.cpp`.

**Where.** `Eigen/src/Spaces/Reset.h`, behind the `Eigen/Spaces` umbrella.

## unit-quaternion

*A unit quaternion space, stored as Eigen stores a quaternion and viewed through a map.*

**Rule.** `UnitQuaternion` stores (x, y, z, w), the order of `Quaternion::coeffs()`, so a state's attitude block is
exactly what a `Quaternion` stores, and its product is Eigen's (Hamilton's). Differences are rotation vectors in the
body frame: $q \boxplus \delta = q \otimes \mathrm{Exp}(\delta)$ and $q \boxminus p = \mathrm{Log}(p^{-1} \otimes q)$
(Solà, Deray, and Atchuthan 2018). `normalize()` keeps $w \ge 0$. The exponential and logarithm are the space's own,
smooth at zero, and its reset Jacobian is the closed-form right Jacobian [covariance-reset]. `UnitQuaternion::map()`
returns a `Map` over a block, always a view, never a copy; a block it can't view, or a plain matrix passed as a
temporary, is a compile error. `map()` is a member of the space, not part of the `Space` concept, since nothing generic
calls it. It is the library's SO(3) [lie-groups].

**Why.** Attitude is the most common state that is not a plain vector, and a user who writes the space must get the
small-angle series, the double cover, and autodiff-safe branches right. Keeping Eigen's storage means the user's
quaternion and the estimator's are one object, not a conversion with an order to get wrong. A map is the only way to
write in place and costs nothing in a debug build; a map that silently became a copy for an expression would carry a
lifetime rule that held only sometimes.

**Rejected.** (w, x, y, z), which matches Eigen's four-scalar constructor but not its storage, so every view would be a
copy. A `toEigen()` that returns a copy, and a `fromEigen()`: writing through the map does both. A generic
`view(space, block)` customization point, which waits for a second space that wants a typed form (S², SE(3),
geodetic).

**Where.** `Eigen/src/Geometry/UnitQuaternion.h`.

## lie-groups

*Lie groups are spaces; navigation is domain models written on them, elsewhere.*

**Rule.** The Lie groups a state lives on are spaces: SO(3) as `UnitQuaternion` [unit-quaternion], and $SE(3)$ and
$SE_2(3)$ beside it when an invariant filter needs them, each with the side its increment is applied on. Inertial
mechanization, IMU preintegration, GNSS, magnetometers, gravity, and geodetic frames are domain models written on these
spaces, in a navigation library, not here.

**Why.** A group is geometry, not navigation. An optimizer, a graph library, a controller, and a sampler need $SE(3)$
and none of them navigates. The quaternion is also the one space whose differences have another size than its values,
so the tests run the manifold paths through it.

**Rejected.** The groups in a navigation project. Leaving every group to the user, which hands each one the small-angle
series and Jacobians that [unit-quaternion] exists to spare them.

**Where.** `Eigen/src/Geometry/UnitQuaternion.h`; the pose groups are not built.

## linear-space

*The plain space is named Linear, for its structure rather than its set.*

**Rule.** The space whose values are any real vectors, with $x \boxplus \delta = x + \delta$ and
$a \boxminus b = a - b$, is `Linear<N>`.

**Why.** What generic code uses about that space is its structure, not its set: `isLinear<>`, `hasPlainValues<>`, and
the one-pass weighted moments depend on `plus()` and `minus()` being addition and subtraction, and $\mathbb{R}^n$ with
a `plus()` of its own would be the same set and no longer `Linear`. "Linear space" is the standard synonym for a vector
space.

**Rejected.** `R<3>`. `Real<3>` (reads as a number type, and collides with `Real` in `NumTraits`). `Reals<3>` and
`RealSpace<3>`. `Euclidean<3>`, the runner-up, which names the metric estimators assume but makes no claim about the
operations, which are what the code checks.

**Where.** `Spaces/Coordinates.h`.

## no-bounded-coordinates

*There is no bounded coordinate type.*

**Rule.** The coordinate types are `Linear` and `Periodic`. A quantity that can't be negative, such as a range or a
speed, is `Linear`. Where an estimator must keep it nonnegative, the user estimates the vector it is the length of and
measures the length, or estimates its log. A distribution with a bounded support states the support itself.

**Why.** A coordinate that reflects a value back across a bound makes $x \boxplus \delta$ nonlinear, while every
Gaussian operation treats it as the identity: samples were folded and the density was not, so near a bound they
disagreed (with mean 0.2 and $\sigma = 1$, 42% of the density's mass lay below zero, and the samples' mean was 0.817),
and a reflected mean kept the old sign of its cross-covariances. Reflection is the physical answer only for the length
of a vector that passes through the origin, and there the standard model is the vector. A bounded type can be added
later without breaking any user; changing what a released one means could not.

**Rejected.** `Bounded<{.lower, .upper}>` and `Magnitude<>`, removed for the reasons above. Folding the density to
match the samples, which breaks the quadratic form that conditionals and updates rely on.

**Where.** `Spaces/Coordinates.h`.

## several-resizable-blocks

*A composite may have several resizable blocks, and only a space with at most one takes its sizes from a length.*

**Rule.** A `Composite` may have any number of resizable blocks, in any position and at any depth, so a state can hold
several growing sets (`Composite<Pose, Tracks, Landmarks>`). A block after the first resizable one has its offset only
at run time. Only a space with at most one resizable block takes its sizes from a length: it has `resize(n)`. A space
with several is built with its counts, and grows and shrinks one block at a time (`internal::growBlock()`,
`internal::shrinkBlock()`). A resizable block inside a `Repeated` stays rejected: ragged data flattens into sibling
sets (`Composite<Fleet, Sensors>`), with each sensor's vehicle kept outside the state as a known association.

**Why.** A state with two growing sets could not be written otherwise. Inferring the split from a length can't be made
safe: the right total split the wrong way (two tracks and five landmarks against data for three and three) would go
unnoticed, so with several blocks the counts must come from the caller's space, and the check is at compile time.

**Rejected.** A resizable block inside a repeated one, which would need a run-time index per repeated block crossed, a
space per element, and index vectors in place of strided sequences everywhere, for a case that sibling sets already
express.

**Where.** `Spaces/Composite.h`, `Spaces/Subspace.h`.

## coordinate-parameters

*A coordinate type's parameters are template arguments, and an interval is spelled by its ends.*

**Rule.** `Periodic<{.lower, .upper}, N>` takes its half-open interval as a structural template argument, so a tag
needs no constructor, a fixed-size coordinate type is an empty type, and the parameters are part of the type. Both ends
must be given; an omitted one is NaN, which a `static_assert` rejects. The common angles are named by the ends of their
interval, as MATLAB's `wrapToPi` family is: `RadiansPi`, `Radians2Pi`, `Degrees180`, `Degrees360`. They name behavior;
quantities such as a heading are the user's tags. Wrapping keeps a value inside its interval exactly, for every input.
Parameters are `double` whatever the scalar, cast where used. The interval becomes a quantity when the unit joins the
leaf ([mandatory-units](decisions.md#mandatory-units)).

**Why.** Parameters in the type cost no storage, and a measurement written for a heading in degrees can't be handed a
state in radians. An interval written by its ends shows at the call site what it is. Values known only at run time
(DFT bins of a configured size is the one case known) are left to a space the user writes.

**Rejected.** Constructor parameters, which made every tag write `using Periodic::Periodic;` and made a fixed space
carry state. A default interval for `Periodic<>`: `Periodic<{.upper = 360.0}>` would hide its lower end. The spelling
`{.period, .center}`: `{.period = 360.0}` silently meant $[-180, 180)$. Per-repetition parameters in a `Repeated`
block: no case needs them. A macro to skip the exact-bounds checks: they run only for values outside the interval, and
translation units that disagreed on the macro would break the one-definition rule. SIMD wrapping, which waits for a
benchmark that shows normalization matters.

**Where.** `Spaces/Coordinates.h`; `periodic_wraps_values_near_far_seams` in `test/spaces.cpp`.

## difference-spaces

*A space has at most one normalize(); its differences live in a difference space; updates and residuals are plus() and
minus(), with defaults.*

**Rule.** A space may define `normalize()` for its values, `difference()` naming the space its differences live in
(itself if absent), and its own `plus()` and `minus()`; a space with any of them must have `difference()`, and one
whose differences have another size must have both operations. Generic code calls only the free functions, which
default to `normalizeValues(space, x + d)` and `normalizeDifferences(space, a - b)`. Linearity is derived:
`hasPlainValues<S>`, `hasPlainDifferences<S>`, and `isLinear<S>` come from which members exist. A space says what it
does through members, which its tags inherit.

**Why.** Each space writes only what it does: an unwrapped heading has a difference space and no `normalize()`. A
$\boxplus$ that is not addition up to canonical form, as on a rotation or a geodetic position, needs the base point,
which a one-argument `normalize()` never sees. A derived flag can't be wrong; a declared one could skip wrapping
silently.

**Rejected.** A pair `normalizeValues()` and `normalizeDifferences()` that every non-linear space defined together,
which made one-sided spaces write an empty function. Flags marking a space as a value or a difference space: a
difference space can differ in more than a flag. Typed difference vectors wrapping Eigen's expressions, which the shape
parameters now do inside them ([spaces-are-shape](decisions.md#spaces-are-shape)). `plus()` and `minus()` with no
normalize, which would hide canonicalizing a value from outside behind `plus(x, 0)`. A declared `isLinear` flag. A
specializable `SpaceTraits<S>`: a specialization doesn't reach a tag, and it would bring back declared flags.

**Where.** `Spaces/Space.h`.

## tag-addressing

*Blocks are addressed by tag path, never by type or by index.*

**Rule.** A path names blocks by tag: a type derived from a space. A library space type, such as `RadiansPi<>`, is
never a path element, even where it is unique. A path of more than one tag is always a `TagPath`, so a template's
argument list is always a list of paths. A tag inside a `Repeated` block is reached only by a path through the block,
and is not counted against the same tag outside it. `idx()` and `diffIdx()` serve raw vectors until a matrix carries
its spaces ([typed-blocks](decisions.md#typed-blocks)).

**Why.** Tags exist to make addressing readable, and a bare type doesn't say what it selects. A path names a block,
which has rows in both the value and the difference space; an index set names rows of one of them, and no longer
addresses the other once their sizes differ, as on a unit quaternion. Counting the inner tags as ambiguous would make
an ownship beside a fleet of targets reachable only through a one-tag path.

**Rejected.** Addressing an untagged block by its unique type. A path as a template pack (`marginal<Ownship,
Position>()`), which looked like a list of two paths elsewhere. An `operator/` on tag instances
(`Ownship{} / Position{}`), which saves five characters and doesn't work where a type is wanted. Index forms kept where
value and difference sizes agree: two ways to do one thing, and code that stops compiling when a block's tangent size
changes.

**Where.** `Spaces/TagPath.h`, `Spaces/Subspace.h`, `Spaces/Indexing.h`.

## tangent-gaussians

*A Gaussian on a non-linear space is a Gaussian in the tangent space at its mean; the wrapped normal is not computed.*

**Rule.** A Gaussian density over a space is $\exp(-\tfrac12 (x \boxminus \mu)^T \Sigma^{-1} (x \boxminus \mu))$
with the usual normalizer. On a `Periodic` coordinate this is valid while $\sigma$ is small against the period:
integrated over $[-\pi, \pi)$, the density has mass 1.000 at $\sigma = 0.5$, 0.998 at $\sigma = 1$, and 0.884 at
$\sigma = 2$. A spread on the order of the period is a distribution that a Gaussian doesn't represent. A narrow
Gaussian on the wrap point is fine, since residuals take the short way round.

**Why.** Efficient execution comes first, and wherever a Gaussian is the right representation the wrapped and
unwrapped densities agree to that precision.

**Rejected.** The wrapped normal, a sum over periods that breaks the quadratic form that conditionals and updates rely
on, as folding a bounded density did [no-bounded-coordinates].

**Where.** `Spaces/Coordinates.h`; the densities themselves are EmrysFilter's.

## space-storage

*A space costs what its state costs.*

**Rule.** A `Composite` stores only its parts with state, and an empty part is one shared instance, so every
fixed-size composite is an empty type. A resizable space stores only its counts. A space whose instance holds nothing,
`FixedSizeSpace` (empty and default-constructible), is named rather than passed: the free functions that take a space
first (`plus`, `minus`, `normalizeValues`, `normalizeDifferences`) also take it as a template argument.

**Why.** The spaces are meant to be as lean as Eigen's fixed-size types: a `std::tuple` of empty leaves added 8 bytes
to a distribution. A per-element copy is only a resizable space's counts, 8 bytes against the kilobytes of a covariance
at its capacity.

**Rejected.** A trailing space defaulted to `S{}`: it still names `S`. Leaving out a resizable space: a function would
build it on every call, from counts it can't always read.

**Where.** `Spaces/Composite.h`, `Spaces/Space.h`.
