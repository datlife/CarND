#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    // measure error of postion and velocity
    rmse << 0,0,0,0;

    // TODO: YOUR CODE HERE

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    if (estimations.size() == 0 ){
        std::cout << "No estimation is found.\n";
        return rmse;
    }
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size()){
        std::cout  << "Size mismatched between estimation and ground truth\n";
        return rmse;
    }

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        // Calculate x - x_true
        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        // (x - x_true)^2
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    //calculate the mean
    rmse = rmse / estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    //TODO: YOUR CODE HERE
    //check division by zero
    float distance = sqrt(px*px + py*py);
    if (distance < 0.00001){
        std::cout << "CalculateJacobian() error - division is zero  at...." << std::endl;
        return Hj;
    }
    //compute the Jacobian matrix
    float vx = x_state(2);
    float vy = x_state(3);

    // Watch : Linearization non-linear models to understand Taylor series first
    // Range Rho
    float d_rho_posx = px / distance;
    float d_rho_posy = py / distance;

    // Angle thea
    float d_theta_posx = - py / (distance*distance);
    float d_theta_posy =   px / (distance*distance);

    // Range Rate:
    float d_rho_rate_posx = py*(vx*py - vy*px)/(distance*distance*distance);
    float d_rho_rate_posy = px*(vy*px - vx*py)/(distance*distance*distance);
    float d_rho_rate_velx = d_rho_posx;
    float d_rho_rate_vely = d_rho_posy;

    // Compute Jacobian matrix
    Hj <<   d_rho_posx,      d_rho_posy,      0,     0,
            d_theta_posx,    d_theta_posy,    0,     0,
            d_rho_rate_posx, d_rho_rate_posy, d_rho_rate_velx, d_rho_rate_vely;

    return Hj;
}
