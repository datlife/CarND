#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    // State Transition Matrix: F_
    // Current State : x_

    float dt = F_(0, 2);
    VectorXd x_prev = x_;

    // Making prediction.
    x_ = F_ * x_;
    // Calculate Acceleration to build better Kinematic Model
//    if (dt > 0.00001) {
//        double delta_vx = x_(2) - x_prev(2);
//        double delta_vy = x_(3) - x_prev(3);
//
//        double x_acc = delta_vx / dt;
//        double y_acc = delta_vy / dt;
//
//        VectorXd control_matrix(4);
//        control_matrix << x_acc * (dt * dt),
//                y_acc * (dt * dt),
//                x_acc * dt,
//                y_acc * dt;
//        x_ = x_ + control_matrix;
//    }
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    // Calculate new measurement
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;

    // Calculate Kalman gain K = P*H / (H*P*Ht + R)
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    K_ = (P_ * H_.transpose()) * S.inverse();

    //new estimate
    x_ = x_ + K_ * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K_ * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

    // Calculate new measurement
    // Note that this is x' , which is predicted values in Predict Step
    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];

    double rho = sqrt(px*px + py*py);
    double theta = atan2(py ,px);
    double rho_dot = 0;
    if (fabs(rho) > 0.0001)
        rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;


    VectorXd H = VectorXd(3);
    H << rho,
         theta,
         rho_dot;


    VectorXd y = z - H;
    // Normalize angle into range [-pi, pi]
    while (y[1] < -M_PI)
        y[1] += 2 * M_PI;
    while (y[1] > M_PI)
        y[1] -= 2 * M_PI;

    // Calculate Kalman gain
    MatrixXd S = H_ * P_ * H_.transpose() + R_; // H_ in this case is Jacbobian Hj
    K_ = (P_ * H_.transpose()) * S.inverse();

    //new estimate
    x_ = x_ + K_ * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K_ * H_) * P_;

}
