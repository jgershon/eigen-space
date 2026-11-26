// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// An invariant extended Kalman filter on typed matrices (design/roadmap.md step 6), without EmrysFilter: a body's
// attitude, velocity, and position on SE_2(3) from an inertial unit and landmarks seen from the body, every matrix in
// units. The error is a step on the right, X = X^ Exp(xi), in radians, metres per second, and metres, whose dynamics
// for an inertial unit are those of Hartley et al. (2020) with the error Barrau and Bonnabel (2017) call
// left-invariant:
//
//   d xi / dt = A xi - w,   A = [ -[w]x    0      0   ]
//                              [ -[a]x  -[w]x    0   ]
//                              [   0      I    -[w]x ],
//
// with w and a the gyroscope's and accelerometer's readings and w their noise. The same filter runs on plain matrices
// beside the typed one; the two must agree to rounding, and the typed one's errors must be consistent with its
// covariance by their normalized squared error.

// Before main.h, whose forbidden-identifier macros would break mp-units' headers.
#include <mp-units/math.h>
#include <mp-units/systems/isq.h>
#include <mp-units/systems/si.h>

#include "main.h"

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/Units>

#include <array>
#include <cmath>
#include <random>

namespace {

namespace si = mp_units::si;
namespace isq = mp_units::isq;

constexpr auto metrePerSecond = si::metre / si::second;
constexpr auto metrePerSecondSquared = si::metre / si::second / si::second;

struct Attitude : QUnitQuaternion<si::radian> {};
struct Velocity : QLinear<isq::velocity, 3> {};
struct Position : QLinear<isq::displacement, 3> {};
struct Body : ExtendedPose<Attitude, Velocity, Position> {};
using State = In<Body, metrePerSecond, si::metre>;
using Step = DifferenceSpace<State>;

using Mean = QVector<State>;
using Error = QVector<Step>;
using Covariance = QMatrix<Step, Step>;
using Transition = QMatrix<Step, Dual<Step>>;

using VelocitySI = In<Velocity, metrePerSecond>;
using PositionSI = In<Position, si::metre>;
using AttitudeStep = QLinear<si::radian, 3>;

// A landmark seen from the body, R^T (l - p).
struct Sighting : QLinear<isq::displacement, 3> {};
using SightingSI = In<Sighting, si::metre>;
using Measurement = QVector<SightingSI>;
using MeasurementCovariance = QMatrix<SightingSI, SightingSI>;
using Jacobian = QMatrix<SightingSI, Dual<Step>>;
using Gain = QMatrix<Step, Dual<SightingSI>>;

using AngularRate = QLinear<si::radian / si::second, 3>;
using SpecificForce = QLinear<metrePerSecondSquared, 3>;

using Vector9d = Matrix<double, 9, 1>;
using Vector10d = Matrix<double, 10, 1>;
using Matrix9d = Matrix<double, 9, 9>;
using Matrix39d = Matrix<double, 3, 9>;

const auto dt = 0.01 * si::second;  // a 100 Hz inertial unit
const auto gyroDensity = 1e-5 * si::radian * si::radian / si::second;
const auto accelDensity = 1e-3 * si::metre * si::metre / (si::second * si::second * si::second);
const auto sightingVariance = 0.01 * si::metre * si::metre;
const QVector<SpecificForce> gravity(0.0, 0.0, -9.81);
const std::array<Vector3d, 3> landmarks{Vector3d{10.0, 0.0, 2.0}, Vector3d{-4.0, 9.0, -1.0}, Vector3d{-3.0, -8.0, 4.0}};

// The typed filter.
struct TypedFilter {
  Mean x;
  Covariance p;

  Matrix3d rotation() const {
    const Vector4d q = x.segment<Attitude>();
    return Quaterniond{q}.toRotationMatrix();
  }

  void predict(const QVector<AngularRate>& gyro, const QVector<SpecificForce>& accel) {
    const Matrix3d turned = rotation();
    const QVector<AttitudeStep> turn = dt * gyro;
    const QVector<SpecificForce> acceleration = turned * accel + gravity;
    const QVector<VelocitySI> velocity = x.segment<Velocity>();
    x.segment<Attitude>() = plus(x.segment<Attitude>(), turn);
    x.segment<Position>() += dt * velocity + (0.5 * dt * dt) * acceleration;
    x.segment<Velocity>() += dt * acceleration;

    // The error dynamics times the step, A dt, block by block: radians per radian, metres per second per radian,
    // seconds where the position follows the velocity.
    Transition m = Transition::Zero();
    m.block<Attitude, Attitude>() = -skew<AttitudeStep, AttitudeStep>(turn);
    m.block<Velocity, Attitude>() = -skew<AttitudeStep>(dt * accel);
    m.block<Velocity, Velocity>() = -skew<VelocitySI, VelocitySI>(turn);
    m.block<Position, Velocity>() = dt * QMatrix<VelocitySI, Dual<VelocitySI>>::Identity();
    m.block<Position, Position>() = -skew<PositionSI, PositionSI>(turn);
    const Transition phi = Transition::Identity() + m + 0.5 * m * m;

    // The readings' noise over the step, in the body frame: the gyroscope's in radians, the accelerometer's in metres
    // per second, as the outer products of their standard deviations.
    const auto gyroSpread = mp_units::sqrt(gyroDensity * dt) * Matrix3d::Identity();
    const auto accelSpread = mp_units::sqrt(accelDensity * dt) * Matrix3d::Identity();
    Covariance q = Covariance::Zero();
    q.block<Attitude, Attitude>() = gyroSpread * gyroSpread.transpose();
    q.block<Velocity, Velocity>() = accelSpread * accelSpread.transpose();
    p = phi * p * phi.transpose() + q;
  }

  void update(const QVector<PositionSI>& landmark, const Measurement& y) {
    const Matrix3d turned = rotation();
    const QVector<PositionSI> position = x.segment<Position>();
    const Measurement predicted = turned.transpose() * (landmark - position);
    // y = R^T (l - p) moves with the attitude's step by [y]x, in metres per radian, and with the position's by -I.
    Jacobian h = Jacobian::Zero();
    h.middleCols<Attitude>() = skew<AttitudeStep>(predicted);
    h.middleCols<Position>() = -QMatrix<SightingSI, Dual<PositionSI>>::Identity();

    const auto noiseSpread = mp_units::sqrt(sightingVariance) * Matrix3d::Identity();
    const MeasurementCovariance n = noiseSpread * noiseSpread.transpose();
    const MeasurementCovariance s = h * p * h.transpose() + n;
    const Gain gain = s.llt().solve(h * p).transpose();
    const Error correction = gain * (y - predicted);
    x = plus(x, correction);
    const Covariance updated = (Transition::Identity() - gain * h) * p;
    p = 0.5 * (updated + updated.transpose());
  }

  // The normalized squared error of the truth: e^T P^-1 e, a plain number.
  double nees(const Mean& truth) const {
    const Error e = minus(truth, x);
    const Matrix<double, 1, 1> value = e.transpose() * p.llt().solve(e);
    return value(0, 0);
  }
};

// The same filter on plain matrices, step for step.
struct PlainFilter {
  Vector10d x;
  Matrix9d p;

  Matrix3d rotation() const { return Quaterniond{Vector4d{x.head<4>()}}.toRotationMatrix(); }

  void predict(const Vector3d& gyro, const Vector3d& accel) {
    const double step = dt.numerical_value_in(si::second);
    const Matrix3d turned = rotation();
    const Vector3d turn = step * gyro;
    const Vector3d acceleration = turned * accel + Vector3d{0.0, 0.0, -9.81};
    const Vector3d velocity = x.segment<3>(4);
    x.head<4>() = plus(UnitQuaternion{}, x.head<4>(), turn);
    x.tail<3>() += step * velocity + (0.5 * step * step) * acceleration;
    x.segment<3>(4) += step * acceleration;

    Matrix9d m = Matrix9d::Zero();
    m.block<3, 3>(0, 0) = -skew(turn);
    m.block<3, 3>(3, 0) = -skew(Vector3d{step * accel});
    m.block<3, 3>(3, 3) = -skew(turn);
    m.block<3, 3>(6, 3) = step * Matrix3d::Identity();
    m.block<3, 3>(6, 6) = -skew(turn);
    const Matrix9d phi = Matrix9d::Identity() + m + 0.5 * m * m;

    Matrix9d q = Matrix9d::Zero();
    q.block<3, 3>(0, 0) = (gyroDensity * dt).numerical_value_in(si::radian * si::radian) * Matrix3d::Identity();
    q.block<3, 3>(3, 3) =
        (accelDensity * dt).numerical_value_in(metrePerSecond * metrePerSecond) * Matrix3d::Identity();
    p = phi * p * phi.transpose() + q;
  }

  void update(const Vector3d& landmark, const Vector3d& y) {
    const Matrix3d turned = rotation();
    const Vector3d predicted = turned.transpose() * (landmark - x.tail<3>());
    Matrix39d h = Matrix39d::Zero();
    h.block<3, 3>(0, 0) = skew(predicted);
    h.block<3, 3>(0, 6) = -Matrix3d::Identity();
    const Matrix3d n = sightingVariance.numerical_value_in(si::metre * si::metre) * Matrix3d::Identity();
    const Matrix3d s = h * p * h.transpose() + n;
    const Matrix<double, 9, 3> gain = s.llt().solve(h * p).transpose();
    const Vector9d correction = gain * (y - predicted);
    x = plus(State{}, x, correction);
    const Matrix9d updated = (Matrix9d::Identity() - gain * h) * p;
    p = 0.5 * (updated + updated.transpose());
  }
};

template <typename Typed>
Map<const Matrix<double, Typed::RowsAtCompileTime, Typed::ColsAtCompileTime>> numbers(const Typed& m) {
  return Map<const Matrix<double, Typed::RowsAtCompileTime, Typed::ColsAtCompileTime>>(m.data());
}

// The body's turning and accelerating, as the prototype's example has it: the gyroscope's reading as given, and the
// accelerometer's the specific force of a gentle acceleration in the world.
Vector3d trueRate(double t) { return Vector3d{0.3 * std::sin(0.7 * t), 0.2 * std::cos(0.5 * t), 0.4}; }
Vector3d trueWorldAcceleration(double t) {
  return Vector3d{std::cos(0.4 * t), std::sin(0.6 * t), 0.3 * std::cos(0.9 * t)};
}

void typed_invariant_ekf_matches_the_plain_one_and_is_consistent() {
  const int runs = 20;
  const int steps = 1000;  // 10 seconds a run
  const int sightEvery = 10;
  const double step = dt.numerical_value_in(si::second);

  // The prior: the heading known to 0.6 rad, the tilt to 0.2 rad, the velocity to 1 m/s, and the position to 2 m.
  const Vector9d deviations{{0.2, 0.2, 0.6, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0}};
  QMatrix<Step, Linear<9>> spread = QMatrix<Step, Linear<9>>::Zero();
  for (Index i = 0; i < 9; ++i) spread(i, i) = deviations(i);  // radians, metres per second, metres, by row
  const Covariance prior = spread * spread.transpose();
  Vector10d start = Vector10d::Zero();
  start.head<4>() = UnitQuaternion::identity();

  std::mt19937_64 rng(24);
  std::normal_distribution<double> normal;
  const auto gaussian = [&](Index n) {
    VectorXd z(n);
    for (Index i = 0; i < n; ++i) z(i) = normal(rng);
    return z;
  };
  const double gyroSd = std::sqrt(gyroDensity.numerical_value_in(si::radian * si::radian / si::second) / step);
  const double accelSd =
      std::sqrt(accelDensity.numerical_value_in(metrePerSecondSquared * metrePerSecondSquared * si::second) / step);
  const double sightingSd = std::sqrt(sightingVariance.numerical_value_in(si::metre * si::metre));

  double neesSum = 0;
  int neesCount = 0;
  double worstDifference = 0;
  for (int run = 0; run < runs; ++run) {
    TypedFilter typed{Mean{}, prior};
    Map<Vector10d>(typed.x.data()) = start;
    PlainFilter plain{start, numbers(prior)};
    Vector10d truth = plus(State{}, start, (deviations.asDiagonal() * gaussian(9)).eval());

    for (int k = 0; k < steps; ++k) {
      const double t = k * step;
      const Matrix3d truthRotation = Quaterniond{Vector4d{truth.head<4>()}}.toRotationMatrix();
      const Vector3d rate = trueRate(t);
      const Vector3d force = truthRotation.transpose() * (trueWorldAcceleration(t) - Vector3d{0.0, 0.0, -9.81});
      // The truth moves as the filters' means do, under the true readings.
      const Vector3d velocity = truth.segment<3>(4);
      const Vector3d acceleration = trueWorldAcceleration(t);
      truth.head<4>() = plus(UnitQuaternion{}, truth.head<4>(), (step * rate).eval());
      truth.tail<3>() += step * velocity + (0.5 * step * step) * acceleration;
      truth.segment<3>(4) += step * acceleration;

      const Vector3d gyroReading = rate + gyroSd * gaussian(3);
      const Vector3d accelReading = force + accelSd * gaussian(3);
      QVector<AngularRate> gyro;
      Map<Vector3d>(gyro.data()) = gyroReading;
      QVector<SpecificForce> accel;
      Map<Vector3d>(accel.data()) = accelReading;
      typed.predict(gyro, accel);
      plain.predict(gyroReading, accelReading);

      if (k % sightEvery == sightEvery - 1) {
        const Vector3d landmark = landmarks[std::size_t(k / sightEvery) % landmarks.size()];
        const Matrix3d rotationNow = Quaterniond{Vector4d{truth.head<4>()}}.toRotationMatrix();
        const Vector3d sighting = rotationNow.transpose() * (landmark - truth.tail<3>()) + sightingSd * gaussian(3);
        QVector<PositionSI> typedLandmark;
        Map<Vector3d>(typedLandmark.data()) = landmark;
        Measurement typedSighting;
        Map<Vector3d>(typedSighting.data()) = sighting;
        typed.update(typedLandmark, typedSighting);
        plain.update(landmark, sighting);
      }
      worstDifference = (std::max)(worstDifference, (numbers(typed.x) - plain.x).cwiseAbs().maxCoeff());
      worstDifference = (std::max)(worstDifference, (numbers(typed.p) - plain.p).cwiseAbs().maxCoeff() /
                                                        (std::max)(1.0, plain.p.cwiseAbs().maxCoeff()));
      if (k >= steps / 2 && k % 50 == 49) {
        Mean typedTruth;
        Map<Vector10d>(typedTruth.data()) = truth;
        neesSum += typed.nees(typedTruth);
        ++neesCount;
      }
    }
  }
  // The typed filter computes what the plain one does, to rounding.
  VERIFY(worstDifference < 1e-9);
  // Its errors are consistent with its covariance: the average of e^T P^-1 e is the error's 9 coordinates.
  const double averageNees = neesSum / neesCount;
  std::cout << "average NEES " << averageNees << " over " << neesCount << " samples; largest typed-plain difference "
            << worstDifference << "\n";
  VERIFY(averageNees > 6.0 && averageNees < 13.0);
}

}  // namespace

EIGEN_DECLARE_TEST(invariant_ekf) { CALL_SUBTEST_1(typed_invariant_ekf_matches_the_plain_one_and_is_consistent()); }
