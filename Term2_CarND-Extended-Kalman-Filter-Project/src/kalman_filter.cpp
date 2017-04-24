#include "kalman_filter.h"
#include <iostream>
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
    x_prev_ = x_;
    // Making prediction.
    x_ = F_ * x_;
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
    //AddAcceleration(); // should have been calculated in Q

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

    if (fabs(rho) > 0.0001) // avoid division by zero
        rho_dot = (px*vx + py*vy)/rho;


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

void KalmanFilter::AddAcceleration(){

    float dt = F_(0, 2);
    if (dt > 0.00001) {
        // Calculate acceleration in (x,y) plane
        double vx_crr = x_(2);
        double vy_crr = x_(3);
        double ax = (vx_crr - x_prev_(2))/dt;
        double ay = (vy_crr - x_prev_(3))/dt;

        VectorXd acc = VectorXd(2);
        acc << ax, ay;

        // Control Matrix B
        MatrixXd B = MatrixXd(4, 2);
        B << 0.5*dt*dt, 0,
                0,  0.5*dt*dt,
                dt, 0,
                0, dt;
        std::cout <<"OLD X = \n" << x_ << std::endl;
        x_ = x_ + B*acc;
    }
}

