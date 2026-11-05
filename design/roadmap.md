# Roadmap

The work in order, written for contributors; what the fork does and why is in
[decisions.md](decisions.md). Kept as EmrysFilter's roadmap: only work not done, a step ends with
"Done when", step numbers are never reused, and a finished step becomes a row of the Done table.

## 0. C++20 baseline — in progress

[cxx20-baseline]. The fork's first commit, so every later step is written against it.

- [x] Raise `cxx_std_14` and `CMAKE_CXX_STANDARD` in `CMakeLists.txt`, the version check in
      `Macros.h`, the SYCL CMake modules, and the CI jobs and scripts that pin an older standard.
- [x] Rewrite `AGENTS.md` rule 3 and the C++14 section of `.agents/conventions.md`; drop the
      C++14 checks from `scripts/check_style.py`.
- [x] Remove the deprecated Qt conversions from `Geometry/Transform.h` [q-names].
- [ ] A focused set of existing tests built and passed under the new default with clang.

Done when: the focused set passes on clang here, and nothing in the tree names C++14 as a
requirement. The whole suite on GCC, Clang, and MSVC runs when a CI machine is available.

## 1. The spaces of EmrysFilter — not started

[spaces-are-shape] (the space types only, not yet the shape parameter), [continuous-spaces-only],
[autodiff-supported], [public-space-helpers], [bibliography-page]. The continuous Space layer of
EmrysFilter copied into Eigen, adapted to its namespace, macros, style, and test framework, so
that EmrysFilter can later take its spaces from here and delete its own.

- [ ] `Eigen/src/Core/Spaces/`: Space.h (the concept, plus(), minus(), the dimension constants),
      Coordinates.h (`Linear`, `Periodic`, the named angles), Composite.h (`Composite`,
      `Repeated`, the part storage), TagPath.h, Subspace.h, and the index-set and row insertion
      and erasure helpers from Indexing.h. `EMRYS_FILTER_ASSERT` becomes `eigen_assert`, the
      no-unique-address macro becomes `EIGEN_NO_UNIQUE_ADDRESS`, `Internal` becomes `internal`.
- [ ] `contrib/Eigen/AutoDiff` promoted to `Eigen/AutoDiff`, with the old path forwarding.
- [ ] `Eigen/Spaces`, the umbrella that includes Core, LU, and AutoDiff: Reset.h's space part
      (hasIdentityReset, the resetJacobian() protocol, forEachResetBlock, the free
      resetJacobian()) and the automatic-differentiation fallback.
- [ ] `UnitQuaternion` in `Eigen/src/Geometry/`.
- [ ] Tests: EmrysFilter's `tests/Spaces/*.cpp` ported to `EIGEN_DECLARE_TEST` and `VERIFY`, its
      `compile_errors.cpp` cases as one failtest each.
- [ ] `doc/Bibliography.dox` seeded with the six keys the headers cite; the fourteen Spaces
      decisions of EmrysFilter brought into decisions.md.
- [ ] Which helpers EmrysFilter calls from outside its Spaces layer are public, each named for
      what it is.

Done when: the ported tests and failtests pass, the Doxygen build has no warning from the new
headers, and EmrysFilter compiles against the fork with its own `Spaces/` directory deleted,
except Discrete.h.

## 2. Units and duals — not started

[mandatory-units], [dual-spaces]. The unit on every leaf, the dual, and mp-units available to the
tests.

- [ ] The unit concept and `Eigen::dimensionless`; the unit parameter on `Linear` and `Periodic`,
      whose interval becomes a quantity.
- [ ] `Dual<S>`: inverts units, distributes over `Composite` and `Repeated`, collapses on itself.
- [ ] mp-units for the tests: `find_package(mp-units CONFIG)` with a FetchContent fallback on a
      pinned release, an option to forbid the fetch, `cmake_minimum_required` raised to 3.25,
      and the backend reported as the other optional ones are. The unit-concept test instantiates
      with mp-units units beside the built-in ones.

Done when: the unit and dual tests pass with and without mp-units found, and a product of two
leaves whose units cancel is dimensionless.

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
