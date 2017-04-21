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
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    H_ << 1, 0, 0, 0,
          0, 1, 0, 0;
    // Kalman Gain
    VectorXd K_;

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

void KalmanFilter::UpdateEKF(const VectorXd &z, const Eigen::MatrixXd & Hj) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

    // Calculate new measurement
    // Note that this is x' , which is predicted values
    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];

    double theta = py/py;
    if (theta > M_PI){
        do{
            theta = theta - 2*M_PI;
        }while(theta > M_PI);
    }
    else if (theta < -M_PI){
        do{
            theta = theta + 2*M_PI;
        }while(theta < -M_PI);
    }

    VectorXd H = VectorXd(3);
    H << sqrt(px*px + py*py),
         atan(theta),
         (px*vx + py*vy)/sqrt(px*px + py*py);

    VectorXd y = z - H;
    // Kalman Gain
    VectorXd K_;
    // Calculate Kalman gain K = P*H / (H*P*Ht + R)
    MatrixXd S = Hj * P_ * Hj.transpose() + R_;
    K_ = (P_ * Hj.transpose()) * S.inverse();

    //new estimate
    x_ = x_ + K_ * y;
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K_ * Hj) * P_;
}
