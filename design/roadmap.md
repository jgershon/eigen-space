# Roadmap

The work in order, written for contributors; what the fork does and why is in
[decisions.md](decisions.md) and [spaces-decisions.md](spaces-decisions.md). Kept as EmrysFilter's
roadmap: only work not done, a step ends with "Done when", step numbers are never reused, and a
finished step becomes a row of the Done table.

## 1a. EmrysFilter on the fork's spaces — not started

[public-space-helpers]. The follow-up to step 1 that only EmrysFilter can drive: it takes its spaces
from the fork and deletes its own `Spaces/` directory, except Discrete.h [continuous-spaces-only].

- [ ] Each helper EmrysFilter calls from outside its Spaces layer is public under a name for what it
      is, or EmrysFilter stops calling it: the single-row-aware plain matrix alias
      (`internal::MatrixOf`), the symmetric-block readers (`internal::lowerSymmetricBlock` and its
      kin), the resolved coordinate selection (`internal::Coordinates`), and the information-form
      reset (`internal::resetInfoAt`).
- [ ] EmrysFilter includes `Eigen/Spaces` and `Eigen/Geometry` in place of its `Spaces/` headers,
      with `Emrys::Internal` spellings of the ported helpers replaced.

Done when: EmrysFilter's tests pass against the fork with its own `Spaces/` directory reduced to
Discrete.h, and nothing in EmrysFilter names `Eigen::internal`.

## 3. Shape on spaces — not started

[spaces-are-shape], [q-names], [space-first], [dual-spaces], [typed-blocks],
[decomposition-boundaries]. The refactor that makes a space the primary shape parameter.

- [ ] `QMatrix` and `QArray` as the primary templates; `Matrix`, `Array`, `Vector`, `RowVector`,
      and the typedefs as aliases; `RowSpace` and `ColSpace` in traits, with the integer constants
      derived. Every existing test passes unchanged.
- [ ] The plain-type helpers in `XprHelper.h`, `Block`, `Map`, `Ref`, `Replicate`, `Reshaped`,
      and the reductions computing spaces.
- [ ] The space rules in `CwiseBinaryOp`, `Product`, `Transpose`, `Inverse`, `Solve`, `Redux`,
      `VectorwiseOp`, and `Dot`, each with a failtest.
- [ ] Tag addressing: `m(Path{}, Path{})`, `m.rows<Path>()`, `m.block<RowPath, ColPath>()`, the
      uniform-axis rule for blocks by offset, and `untyped()`; idx() and diffIdx() go.
- [ ] The factor spaces of Cholesky, LU, QR, SVD, and the eigensolvers.
- [ ] A compile-time measurement: a fixed set of tests (`product_large`, `basicstuff`, a
      decomposition) timed before and after.
- [ ] A fixed-size Kalman example whose every matrix is typed.

Done when: `buildtests` and `check` pass with no test changed, the failtests fail for the right
reason, the Kalman example matches the same filter on plain matrices to rounding, and the
compile-time measurement is recorded here with any regression explained.

## 4. mp-units — not started

- [ ] `Eigen/MPUnitsSupport`: mp-units units as the unit parameter, `one` mapped onto
      `dimensionless`, and `m.at<RowPath, ColPath>()` returning an mp-units `quantity`.
- [ ] EmrysFilter on `QMatrix`, in its own repository: a `GaussianParams<Space>` whose mean and
      covariance are typed by the space it already names.

Done when: the Kalman example of step 3 is written with mp-units quantities throughout, and an
EmrysFilter radar example types its state, measurement, and gain.

## Done

| Step | What | Decisions |
|---|---|---|
| 0 | C++20 baseline | [cxx20-baseline], [q-names] |
| 1 | The continuous spaces of EmrysFilter in Core, the reset in `Eigen/Spaces`, `UnitQuaternion` in Geometry, AutoDiff promoted, the bibliography page | [continuous-spaces-only], [autodiff-supported], [bibliography-page], the thirteen in spaces-decisions.md |
| 2 | The unit on every leaf (`QLinear`, `QPeriodic`, the Q angles, with the plain names as dimensionless aliases), the `Unit` concept and `dimensionless`, `Dual`, `unitAt()` and `unitOf()`, and mp-units fetched for the tests | [mandatory-units], [dual-spaces] |
