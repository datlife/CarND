#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "Dense"
#include "measurement_package.h"
#include "tracking.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define INPUT_FILE "../obj_pose-laser-radar-synthetic-input.txt"

MatrixXd CalculateJacobian(const VectorXd& x_state);
VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth);
int main() {

    /*******************************************************************************
     *  Set Measurements															 *
     *******************************************************************************/
    vector<MeasurementPackage> measurement_pack_list; // a list of measurement from input file

    // hardcoded input file with laser and radar measurements
    string in_file_name_ = INPUT_FILE;

    // Open the file
    ifstream in_file(in_file_name_.c_str(),std::ifstream::in);
    if (!in_file.is_open()) {
        cout << "Cannot open input file: " << in_file_name_ << endl;
    }

    string line;
    // set i to get only first 3 measurements
    int i = 0;
    while(getline(in_file, line) && (i<=3)){
        MeasurementPackage meas_package;
        istringstream iss(line);
        string sensor_type;
        iss >> sensor_type;	//reads first element from the current line
        long timestamp;
        if(sensor_type.compare("L") == 0){	// Laser measurement
            //read measurements
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = VectorXd(2);
            float x;
            float y;
            iss >> x;
            iss >> y;
            meas_package.raw_measurements_ << x,y;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }
        else if(sensor_type.compare("R") == 0){
            //Skip Radar measurements
            continue;
        }
        i++;
    }

    //Create a Tracking instance
    Tracking tracking;

    //call the ProcessingMeasurement() function for each measurement
    size_t N = measurement_pack_list.size();
    for (size_t k = 0; k < N; ++k) {	//start filtering from the second frame (the speed is unknown in the first frame)
        tracking.ProcessMeasurement(measurement_pack_list[k]);
    }

    if(in_file.is_open()){
        in_file.close();
    }
    return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state){
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    //TODO: YOUR CODE HERE
    //check division by zero
    float distance = sqrt(px*px + py*py);
    if (distance < 0.00001){
        cout << "CalculateJacobian() error - division is zero  at...." <<endl;
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
    Hj << d_rho_posx,      d_rho_posy,      0,     0,
          d_theta_posx,    d_theta_posy,    0,     0,
          d_rho_rate_posx, d_rho_rate_posy, d_rho_rate_velx, d_rho_rate_vely;

    return Hj;
}
VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth){
    VectorXd rmse(4);
    // measure error of postion and velocity
    rmse << 0,0,0,0;

    // TODO: YOUR CODE HERE

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    if (estimations.size() == 0 ){
        cout << "No estimation is found.\n";
        return rmse;
    }
    //  * the estimation vector size should equal ground truth vector size
    if (estimations.size() != ground_truth.size()){
       cout  << "Size mismatched between estimation and ground truth\n";
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