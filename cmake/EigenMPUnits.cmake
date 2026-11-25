# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# mp-units for the tests of the spaces with units (design/decisions.md [mp-units-dependency]). Eigen/Units includes it;
# Eigen/Core does not, so the tests of the plain spaces and everything else build without it.
# An installed package is preferred; without one, and unless EIGEN_FETCH_MPUNITS is off, the pinned release is
# fetched and configured header-only with no transitive dependencies: contracts off (no gsl-lite) and std::format on
# (no fmt). Defines MPUNITS_FOUND and the imported target mp-units::mp-units.

include(FetchContent)

set(EIGEN_MPUNITS_VERSION 2.5.0)

if(EIGEN_FETCH_MPUNITS)
  # The cache entry wins over mp-units' own set(... CACHE STRING ...), and the normal variables over its option()s
  # under CMP0077, which mp-units' cmake_minimum_required(3.25) sets to NEW.
  set(MP_UNITS_API_CONTRACTS NONE CACHE STRING "mp-units contract checking (Eigen tests: none)")
  set(MP_UNITS_API_STD_FORMAT ON)
  set(MP_UNITS_BUILD_INSTALL OFF)
  set(MP_UNITS_BUILD_CXX_MODULES OFF)
  FetchContent_Declare(mp-units
    URL https://github.com/mpusz/mp-units/archive/refs/tags/v${EIGEN_MPUNITS_VERSION}.tar.gz
    URL_HASH SHA256=a6bd48bee699f11f0ed5b04b8c5006d15f76e6d898e058db3880554d2e47a400
    SOURCE_SUBDIR src
    EXCLUDE_FROM_ALL
    SYSTEM
    FIND_PACKAGE_ARGS ${EIGEN_MPUNITS_VERSION} CONFIG
  )
  FetchContent_MakeAvailable(mp-units)
else()
  find_package(mp-units ${EIGEN_MPUNITS_VERSION} CONFIG QUIET)
endif()

if(TARGET mp-units::mp-units)
  set(MPUNITS_FOUND TRUE)
else()
  set(MPUNITS_FOUND FALSE)
endif()
