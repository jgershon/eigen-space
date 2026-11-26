# Roadmap

The work in order, written for contributors; what the fork does and why is in
[decisions.md](decisions.md) and [spaces-decisions.md](spaces-decisions.md). Kept as EmrysFilter's
roadmap: only work not done, a step ends with "Done when", step numbers are never reused, and a
finished step becomes a row of the Done table.

The fork's first goal was to show that units in Eigen's own types are feasible; step 3's slice
showed they are [units-feasible]. EmrysFilter on typed matrices is where the work leads, not where
it starts: mp-units (step 4), a typed invariant EKF on the fork alone (step 6), the rest of
the refactor, and EmrysFilter follow the slice.

## 6. A typed invariant EKF — in progress

[additive-groups], [lie-groups], [heterogeneous-results], [typed-blocks], [decomposition-boundaries].
The fork's next goal: an invariant extended Kalman filter (Barrau and Bonnabel 2017) for a body's
attitude, velocity, and position from an inertial unit and landmarks seen from the body, written on
the fork's typed matrices alone, without EmrysFilter, with every matrix in units. The state is
$SE_2(3)$ (Barrau and Bonnabel 2017; Hartley et al. 2020) and increments apply on the right,
$X = \hat{X} \cdot \mathrm{Exp}(\xi)$, as EmrysFilter's `groups-prototype` decided; the error
$\xi$ is in radians, metres per second, and metres. It needs part of step 5, done first and only as
far as the filter does.

- [x] The group operations of `groups-prototype`, ported with its `ExtendedPose` [lie-groups]:
      `LieGroup`, `Rotation`, `compose()`, `inverse()`, `identity()`, `exp()`, `log()`,
      `adjoint()`, `rightJacobian()`, and `leftJacobian()`, on plain coordinates given the space and
      typed given the group or a typed operand, with `plus()` and `minus()` typed too;
      `QUnitQuaternion<Q>` with its differences in radians, and the group's members on it. Its
      tests, checked against homogeneous matrices and the matrix exponential, and SE_2(3) in
      radians, metres per second, and metres, with a step in feet converted. A bound space's
      difference keeps the difference's own quantities where the space had them. AutoDiff's global
      functions deduce their whole argument, so `exp<Pose>(d)` does not instantiate
      `AutoDiffScalar<Pose>`, and a periodic value wraps under AutoDiff.
- [ ] Blocks by tag, from step 5's tag addressing: `x(Path{})`, `m.block<RowPath, ColPath>()`, and
      their writable forms, typed by the subspaces named, enough to assemble the error dynamics
      and the measurement Jacobian block by block and to read the state's parts.
- [ ] The typed `Identity()` and `Zero()` of a matrix from one space to another, including the
      identity block from velocity to the time derivative of position, whose entries are one.
- [ ] `skew<ColSpace>(v)`: the matrix of $\delta \mapsto v \times \delta$ with rows in $v$'s
      space and columns dual to `ColSpace`. mp-units converts rad x m to m only explicitly (rad/s x
      m to m/s implicitly), so naming the column space, the attitude's radians, is that explicit
      step: $\partial (R^\top (l - p)) / \partial \phi$ is in metres per radian.
- [ ] A typed `llt()` and its `solve()` on a symmetric positive-definite matrix from step 5's factor
      spaces, for the innovation covariance $S$ and the gain $K = P H^\top S^{-1}$, and for the
      consistency statistic $e^\top P^{-1} e$, dimensionless.
- [ ] The error dynamics $A$ with rows in the difference space per second, its discretization
      $\Phi = I + A\,\Delta t + \tfrac12 (A\,\Delta t)^2$ with $\Delta t$ an mp-units
      quantity, and the process noise from the gyroscope's and the accelerometer's noise densities,
      also quantities.
- [ ] The filter, as a test: it simulates a body from a heading known to 0.6 rad, as the
      `groups-prototype` example does, runs the filter on typed matrices and the same filter on
      plain ones, and checks that they agree to rounding and that the filter's errors are
      consistent with its covariances by their normalized squared error. The gyroscope's bias is
      left out.

Done when: the filter's test passes under clang with no plain test changed, the failtests of each
item fail for the right reason, and a failtest shows the filter's update with a Jacobian in the
wrong units (metres where the attitude's block needs metres per radian) not compiling.

## 5. Shape on spaces, the rest — not started

[spaces-are-shape], [dual-spaces], [typed-blocks], [decomposition-boundaries]. Every expression
and decomposition the slice did not reach.

- [ ] `Block`, `Map`, `Ref`, `Replicate`, `Reshaped`, and the reductions computing spaces.
- [ ] The space rules in `Inverse`, `Solve`, `Redux`, `VectorwiseOp`, and `Dot`, each with a
      failtest.
- [ ] Tag addressing: `m(Path{}, Path{})`, `m.rows<Path>()`, `m.block<RowPath, ColPath>()`, the
      uniform-axis rule for blocks by offset, and `untyped()`; idx() and diffIdx() go.
- [ ] The factor spaces of Cholesky, LU, QR, SVD, and the eigensolvers.
- [ ] A fixed-size Kalman example whose every matrix is typed, beside step 6's filter.

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
