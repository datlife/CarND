#include "kalman_filter.h"

KalmanFilter::KalmanFilter() {
}

KalmanFilter::~KalmanFilter() {
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

    // Calculate new measurement
    VectorXd z_pred = H_ * x_;

    // Calculate Kalman gain K = P*H / (H*P*Ht + R)
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    K_ = (P_ * H_.transpose()) * S.inverse();
    //new estimate
    x_ = x_ + K_ * (z - z_pred);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K_ * H_) * P_;
}

