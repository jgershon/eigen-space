# Roadmap

The work in order, written for contributors; what the fork does and why is in
[decisions.md](decisions.md) and [spaces-decisions.md](spaces-decisions.md). Kept as EmrysFilter's
roadmap: only work not done, a step ends with "Done when", step numbers are never reused, and a
finished step becomes a row of the Done table.

The fork's first goal was to show that units in Eigen's own types are feasible; step 3's slice
showed they are [units-feasible]. EmrysFilter on typed matrices is where the work leads, not where
it starts: mp-units (step 4), a typed invariant EKF on the fork alone (step 6), the rest of
the refactor, and EmrysFilter follow the slice. What step 6 found and left open is in its row of
the Done table and in [lie-groups].

## 5. Shape on spaces, the rest — not started

[spaces-are-shape], [dual-spaces], [typed-blocks], [decomposition-boundaries]. Every expression
and decomposition the slice did not reach.

- [ ] `Block`, `Map`, `Ref`, `Replicate`, `Reshaped`, and the reductions computing spaces.
- [ ] The space rules in `Inverse`, `Solve`, `Redux`, `VectorwiseOp`, and `Dot`, each with a
      failtest.
- [ ] Tag addressing, beyond step 6's: `m(Path{}, Path{})`, blocks by tag on spaces of run-time
      size, the uniform-axis rule for blocks by offset, and `untyped()`; idx() and diffIdx() go.
- [ ] The factor spaces of Cholesky, LU, QR, SVD, and the eigensolvers.
- [ ] A fixed-size Kalman example whose every matrix is typed, beside step 6's filter
      (`test/invariant_ekf.cpp`).
- [x] Sums and assignments admit one common factor between the axes, so `(sigma^2 * dt) * I`
      assigns into a covariance block in the same entry units ([dual-spaces]).

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
| 3 | The feasibility slice: `QMatrix` and `QArray` primary with `Matrix` and `Array` as aliases, the space rules of sums, products, transposes, and assignments, a typed covariance prediction, and the compile-time measurement; the outcome is go | [spaces-are-shape], [q-names], [space-first], [dual-spaces], [units-feasible] |
| 4 | Units on mp-units: `Eigen/Units`, leaves of quantity specs, units, or references, `In<>` and `bind<>()`, conversion as mp-units converts, results of operands with different tags, coordinate arithmetic on groups under addition only, complex scalars, and quantities as scalars | [mp-units-dependency], [leaf-quantities], [heterogeneous-results], [additive-groups], [complex-scalars] |
| 6 | A typed invariant EKF on SE_2(3), without EmrysFilter: the group operations of `groups-prototype` and `ExtendedPose`, typed; blocks by tag with difference spaces that keep their tags; `skew<ColSpace>(v)` and `skew<RowSpace, ColSpace>(phi)`; a typed `llt()`; and the filter as a test, matching the same filter on plain matrices to 1.8e-14 with an average normalized squared error of 10.8 for 9 coordinates | [lie-groups], [typed-blocks], [decomposition-boundaries], [heterogeneous-results] |
