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
- [ ] The space rules in `CwiseBinaryOp` (sum and difference: equal spaces), `Product` (the inner
      spaces dual to each other), and `Transpose`, each with a failtest for a mismatched unit.
- [ ] A typed covariance prediction, `P = F * P * F.transpose() + Q`, on a position and velocity
      state with units in metres and metres per second, checked against the same computation on
      plain matrices.
- [ ] A compile-time measurement: `basicstuff` and `product_large` timed before and after. After
      the first item, with clang 21 at -O3 and one job: `basicstuff` 29.2 s before, 31.5 s and
      29.6 s after; `product_large` 196.4 s before, 194.9 s and 198.9 s after; no change beyond
      noise. To repeat once the space rules are in.

Done when: a representative subset of the tests passes with no test changed (the slice is an
experiment; the whole suite is step 5's bar), the failtests fail for the right reason, the typed
prediction matches the plain one to rounding, and the compile-time measurement is
recorded here with any regression explained. The outcome, go or no-go with the reasons, is recorded
in decisions.md before step 5 starts.

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

## 4. mp-units — not started

- [ ] `Eigen/MPUnitsSupport`: mp-units units as the unit parameter, `one` mapped onto
      `dimensionless`, and `m.at<RowPath, ColPath>()` returning an mp-units `quantity`.

Done when: the Kalman example of step 5 is written with mp-units quantities throughout.

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
