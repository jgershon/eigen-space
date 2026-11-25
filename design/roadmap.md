# Roadmap

The work in order, written for contributors; what the fork does and why is in
[decisions.md](decisions.md) and [spaces-decisions.md](spaces-decisions.md). Kept as EmrysFilter's
roadmap: only work not done, a step ends with "Done when", step numbers are never reused, and a
finished step becomes a row of the Done table.

The fork's first goal is to show that units in Eigen's own types are feasible. EmrysFilter on typed
matrices is where the work leads, not where it starts: step 3 is a vertical slice that decides
whether the design holds, and the rest of the refactor, mp-units, and EmrysFilter follow it.

## 3. Feasibility slice — in progress

[spaces-are-shape], [q-names], [space-first], [dual-spaces]. The least of the refactor that answers
whether a space can be Eigen's shape parameter: the alias that keeps every existing spelling, the
typing of the operations a covariance prediction needs, and what it costs to compile.

- [x] `QMatrix` and `QArray` as the primary templates; `Matrix`, `Array`, `Vector`, `RowVector`,
      and the typedefs as aliases; the integer constants derived from the spaces, which
      `internal::spaces_of` reports, and the plain-type helpers in `XprHelper.h` building plain
      objects from spaces. Existing tests pass unchanged, which is the check that deduction and
      partial specialization on `Matrix<Scalar, Rows, Cols, ...>` survive the alias. One header
      changed to fit: `AlignedVector3`'s traits, which borrowed a 3-row `Matrix` with room for 4.
- [x] The space rules in `CwiseBinaryOp` (sum and difference: equal spaces), `Product` (the inner
      spaces dual to each other), and `Transpose`, each with a failtest for a mismatched unit.
      Assignment is checked too, since a typed prediction means nothing if its result can land in
      any matrix, and so is a cwise product of two typed operands; `Spaces/Rules.h` compares
      spaces as runs of one unit, so plain operands skip the rules at the cost of one check.
- [x] A typed covariance prediction, `P = F * P * F.transpose() + Q`, on a position and velocity
      state with units in metres and metres per second, checked against the same computation on
      plain matrices: in `test/qmatrix.cpp`, and over mp-units' SI units in
      `test/spaces_units.cpp`.
- [x] A compile-time measurement: `basicstuff` and `product_large` timed before and after. After
      the first item, with clang 21 at -O3 and one job: `basicstuff` 29.2 s before, 31.5 s and
      29.6 s after; `product_large` 196.4 s before, 194.9 s and 198.9 s after; no change beyond
      noise. With the space rules in, timed back to back against the same baseline on a busier
      machine: `basicstuff` 39.3 s before, 33.4 s after; `product_large` 227.8 s before, 232.6 s
      after (+2%, within the run-to-run spread). Plain operands meet the rules through one check
      of their Linear spaces, which is why the cost does not show.

Done when: a representative subset of the tests passes with no test changed (the slice is an
experiment; the whole suite is step 5's bar), the failtests fail for the right reason, the typed
prediction matches the plain one to rounding, and the compile-time measurement is
recorded here with any regression explained. The outcome, go or no-go with the reasons, is recorded
in decisions.md before step 5 starts.

## 4. Units on mp-units — in progress

[mp-units-dependency], [leaf-quantities], [heterogeneous-results], [additive-groups],
[complex-scalars]. Before step 5, since it changes what the slice's rules compare: the units become
mp-units' own, with conversion, and coordinate arithmetic is checked against the space's group.

- [x] `Eigen/Units`: mp-units as the dependency of spaces with units, `internal::units_backend`
      declared in Core and defined there, leaves whose first argument is a quantity spec, a unit, or
      a reference, and `In<>` binding units to tags by dimension or by tag, kept by position; the
      tests and failtests on mp-units' units, in place of the test unit types. Sums and products
      compare runs of compatible quantities in equivalent units, so metres as a unit and as a
      reference agree and nothing yet converts.
- [x] Conversion: sums and differences in the common unit, assignment and construction converting
      into the destination, a product's inner coordinates converted to one factor c, and
      `in<>()`; no conversion generated where the units agree. Two run lists are refined to one
      partition before they are compared, so blocks that fall differently still match. A
      conversion is a coefficient-wise product with a nullary of row and column factors, relabeled;
      a converting sum is evaluated as the sum of its converted operands, and `operator*` converts
      the right operand. Not yet: a conversion across a repeated block of several units, and the
      check that an integer `Scalar` converts by an integral factor.
- [x] Results of operands with different tags: shared tags kept and rebound, the untagged leaves of
      the common quantities otherwise; a product's row space scaled by its inner unit keeps its tags
      for a ratio of units and drops them for a quantity with a dimension.
- [x] Coordinate arithmetic on groups under addition only: the classification of EmrysFilter's
      `groups-prototype` in `Spaces/Group.h`, held by sums, scalings, `+=` and `-=`, and a product's
      inner spaces, with failtests for a difference and a scaling of quaternions, and a covariance
      over the difference space of a pose with a heading. The plain fast path is now dimensionless
      and additive, since a vector of quaternions is dimensionless.
- [ ] The group operations of `groups-prototype`: `LieGroup`, `compose()`, `inverse()`,
      `identity()`, `exp()`, `log()`, and `UnitQuaternion`'s group members. No rule needs them; the
      error of the rule above names them as what to use.
- [x] Complex scalars: `adjoint()` swapping the spaces, and the character of a leaf's spec checked
      against `Scalar`. Negation, the conjugate, `real()`, `imag()`, and `abs()` keep their operand's
      spaces; any other unary operation still has dimensionless ones.
- [x] mp-units quantities as scalars: `dt * v` with `dt` in seconds, `v * dt`, and `x / dt`, which
      multiply the row space by what the quantity measures (`Eigen/src/Units/Scalars.h`).

Done when: the step 3 subset and the failtests pass, the typed prediction of step 3 runs in SI and
in feet with the same values after conversion, and the compile-time measurement of step 3 is
repeated with any regression explained.

Measured after the items above, clang 21 at -O3 and one job, against the same baseline: run in
sequence, `basicstuff` 40.9 s before and 29.8 s after, `product_large` 265.0 s and 182.0 s, a
machine whose load fell between the runs; interleaved, `basicstuff` 27.3 s and 27.1 s before against
27.8 s and 27.2 s after, no change beyond noise. Plain code still meets the rules through one check
of plain spaces. A typed translation unit pays for mp-units and the rules: `units_1`, which includes
Eigen/Units and exercises every rule, compiles in 8.5 s against 2.2 s for the plain `qmatrix_1`.

## 5. Shape on spaces, the rest — not started

[spaces-are-shape], [dual-spaces], [typed-blocks], [decomposition-boundaries]. Every expression
and decomposition the slice did not reach.

- [ ] `Block`, `Map`, `Ref`, `Replicate`, `Reshaped`, and the reductions computing spaces.
- [ ] The space rules in `Inverse`, `Solve`, `Redux`, `VectorwiseOp`, and `Dot`, each with a
      failtest.
- [ ] Tag addressing: `m(Path{}, Path{})`, `m.rows<Path>()`, `m.block<RowPath, ColPath>()`, the
      uniform-axis rule for blocks by offset, and `untyped()`; idx() and diffIdx() go.
- [ ] The factor spaces of Cholesky, LU, QR, SVD, and the eigensolvers.
- [ ] A fixed-size Kalman example whose every matrix is typed.

Done when: `buildtests` and `check` pass with no test changed, the failtests fail for the right
reason, the Kalman example matches the same filter on plain matrices to rounding, and the
compile-time measurement of step 3, repeated, is recorded here with any regression explained.

## 1a. EmrysFilter on the fork's spaces — not started

[public-space-helpers]. The follow-up to step 1 that only EmrysFilter can drive, and the last of the
fork's work: it takes its spaces from the fork and deletes its own `Spaces/` directory, except
Discrete.h [continuous-spaces-only].

- [ ] Each helper EmrysFilter calls from outside its Spaces layer is public under a name for what it
      is, or EmrysFilter stops calling it: the single-row-aware plain matrix alias
      (`internal::MatrixOf`), the symmetric-block readers (`internal::lowerSymmetricBlock` and its
      kin), the resolved coordinate selection (`internal::Coordinates`), and the information-form
      reset (`internal::resetInfoAt`).
- [ ] EmrysFilter includes `Eigen/Spaces` and `Eigen/Geometry` in place of its `Spaces/` headers,
      with `Emrys::Internal` spellings of the ported helpers replaced.
- [ ] EmrysFilter on `QMatrix`, in its own repository: a `GaussianParams<Space>` whose mean and
      covariance are typed by the space it already names.

Done when: EmrysFilter's tests pass against the fork with its own `Spaces/` directory reduced to
Discrete.h, nothing in EmrysFilter names `Eigen::internal`, and an EmrysFilter radar example types
its state, measurement, and gain.

## Done

| Step | What | Decisions |
|---|---|---|
| 0 | C++20 baseline | [cxx20-baseline], [q-names] |
| 1 | The continuous spaces of EmrysFilter in Core, the reset in `Eigen/Spaces`, `UnitQuaternion` in Geometry, AutoDiff promoted, the bibliography page | [continuous-spaces-only], [autodiff-supported], [bibliography-page], the thirteen in spaces-decisions.md |
| 2 | The unit on every leaf (`QLinear`, `QPeriodic`, the Q angles, with the plain names as dimensionless aliases), the `Unit` concept and `dimensionless`, `Dual`, `unitAt()` and `unitOf()`, and mp-units fetched for the tests | [mandatory-units], [dual-spaces] |
