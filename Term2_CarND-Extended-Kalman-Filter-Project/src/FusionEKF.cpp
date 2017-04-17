#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // initializing matrices
    R_laser_ = MatrixXd(2, 2);
    R_radar_ = MatrixXd(3, 3);
    H_laser_ = MatrixXd(2, 4);
    Hj_ = MatrixXd(3, 4);

    //measurement covariance matrix R - laser
    R_laser_ << 0.0225, 0,
              0,      0.0225;

    //measurement covariance matrix R - radar
    R_radar_ << 0.09, 0,      0,
                0,    0.0009, 0,
                0,    0,      0.09;

    //the initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ <<  1, 0, 1, 0,
                0, 1, 0, 1,
                0, 0, 1, 0,
                0, 0, 0, 1;

    /**
    TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
    */
    noise_ax_ = 9.0;
    noise_ay_ = 9.0;

    // Process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    // Measurement Noise Covariance Matrix Z_

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

Tools FusionEKF::getTool() const{ return tools_;}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
        /**
        TODO:
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // first measurement
        cout << "Initializing EKF: " << endl;
        VectorXd init_state(4);

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
          /**
          Convert radar from polar to cartesian coordinates and initialize state.
          */
            // Convert Polar Coordinate (rho, theta, rho_dot) into Cartesian Coordinates
            //init_state = measurement_pack.raw_measurements_;
            // How ?
            double theta = measurement_pack.raw_measurements_(1);
            double rho_x = measurement_pack.raw_measurements_(0)*sin(theta);
            double rho_y = measurement_pack.raw_measurements_(0)*cos(theta)*(-1);
            init_state << rho_x,rho_y, 0 , 0;
        }
        else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
          /**
          Initialize state.
          */
            init_state << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], // init position
                          0, 0;                                                                         // init velocity

        }
        // State Matrix X_
        ekf_.x_ = init_state;
        // State covariance matrix P
        ekf_.P_ = MatrixXd(4, 4);
        ekf_.P_ <<  1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1000, 0,
                    0, 0, 0, 1000;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        previous_timestamp_ = measurement_pack.timestamp_;
        return;
    }
    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

    /**
    TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    */
    //compute the time elapsed between the current and previous measurements
    float dt = (float)((measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0);	//dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // Compute values for process covariance matrix
    float dt_2 = dt*dt;
    float dt_3 = dt*dt_2;
    float dt_4 = dt*dt_3;

    ekf_.Q_ << noise_ax_*(dt_4/4) ,0 ,                 noise_ax_*(dt_3/2), 0,
               0,                  noise_ay_*(dt_4/4), 0,                  noise_ay_*(dt_3/2),
               noise_ax_*(dt_3/2), 0,                  noise_ax_*(dt_2),   0,
               0,                  noise_ay_*(dt_3/2), 0,                  noise_ay_*(dt_2);

    ekf_.Predict();

    /*****************************************************************************
    *  Update
    ****************************************************************************/
    /**
    TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
    */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
        // Convert to Cartesian Coordinate
        double theta = measurement_pack.raw_measurements_(1);
        double rho_x = measurement_pack.raw_measurements_(0)*sin(theta);
        double rho_y = measurement_pack.raw_measurements_(0)*cos(theta)*(-1);


    }
    else {
    // Laser updates
        ekf_.Update()
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
