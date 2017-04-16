#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ground_truth_package.h"
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//============================ Function Prototypes ============================================
void validate_arguments(int argc, char **argv);
void check_files       (ifstream &, string& in_name, ofstream& out_file, string& out_name);
void read_input        (ifstream &, vector<MeasurementPackage> &, vector<GroundTruthPackage> &);
void fuse_data_sensors (ofstream &, vector<MeasurementPackage> &, vector<GroundTruthPackage> &);
void write_output      (ofstream &, const FusionEKF &, const MeasurementPackage &, const GroundTruthPackage &);

// Input format :
// ------------------------------------------------------------------------------------------
// |                   RAW DATA                             |         GROUND TRUTH           |
// ------------------------------------------------------------------------------------------|
// |   LASER   | X POS    | Y POS       | TIME STAMP        |  X POS | Y POS | X VEL | Y VEL | <-- Cartesian Coordinate
// |   RADAR   | DIST RHO | ANGLE THETA | DIST_RATE RHO DOT |  X POS | Y POS | X VEL | Y VEL | <-- Polar Coordinate
// |-----------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Validate arguments
    validate_arguments(argc, argv);

    // Create input/output stream
    string in_file_name_ = argv[1], out_file_name_ = argv[2];
    ifstream in_file_(in_file_name_.c_str(), ifstream::in);
    ofstream out_file_(out_file_name_.c_str(), ofstream::out);

    // Validate input/output file
    check_files(in_file_, in_file_name_, out_file_, out_file_name_);

    vector<MeasurementPackage> measurement_pack_list;
    vector<GroundTruthPackage> gt_pack_list;

    // Read input file (Laser/Radar Measurement) - passed by reference
    read_input(in_file_, measurement_pack_list, gt_pack_list);

    // Fuse Sensors and write to output -- in same loop
    fuse_data_sensors(out_file_, measurement_pack_list, gt_pack_list);

    // close files
    if (out_file_.is_open()) out_file_.close();
    if (in_file_.is_open()) in_file_.close();

    return 0;
}

void validate_arguments(int argc, char **argv) {
    string usage_instructions = "Usage instructions: ";
    usage_instructions += argv[0];
    usage_instructions += " path/to/input.txt output.txt";
    bool has_valid_args = false;

    // make sure the user has provided input and output files
    if (argc == 1) {
        cerr << usage_instructions << endl;
    } else if (argc == 2) {
        cerr << "Please include an output file.\n" << usage_instructions << endl;
    } else if (argc == 3) {
        has_valid_args = true;
    } else if (argc > 3) {
        cerr << "Too many arguments.\n" << usage_instructions << endl;
    }

    if (!has_valid_args) {
        exit(EXIT_FAILURE);
    }
}
void read_input(ifstream &in_file_, vector<MeasurementPackage> &measurement_pack_list, vector<GroundTruthPackage> &gt_pack_list){

    string line;
    // prep the measurement packages (each line represents a measurement at a timestamp)
    while (getline(in_file_, line)) {
        string sensor_type;
        MeasurementPackage meas_package;
        GroundTruthPackage gt_package;
        istringstream iss(line);
        long long timestamp;
        // reads first element from the current line
        iss >> sensor_type;
        if (sensor_type.compare("L") == 0) {
            // LASER MEASUREMENT

            // read measurements at this timestamp
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = VectorXd(2);
            float x;
            float y;
            iss >> x;
            iss >> y;
            meas_package.raw_measurements_ << x, y;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }
        else if (sensor_type.compare("R") == 0) {
            // RADAR MEASUREMENT

            // read measurements at this timestamp
            meas_package.sensor_type_ = MeasurementPackage::RADAR;
            meas_package.raw_measurements_ = VectorXd(3);
            float ro;
            float phi;
            float ro_dot;
            iss >> ro;
            iss >> phi;
            iss >> ro_dot;
            meas_package.raw_measurements_ << ro, phi, ro_dot;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);
        }
        // read ground truth data to compare later
        float x_gt;
        float y_gt;
        float vx_gt;
        float vy_gt;
        iss >> x_gt;
        iss >> y_gt;
        iss >> vx_gt;
        iss >> vy_gt;
        gt_package.gt_values_ = VectorXd(4);
        gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
        gt_pack_list.push_back(gt_package);
    }

}

void fuse_data_sensors( ofstream &out_file_, vector<MeasurementPackage> &measurement_pack_list, vector<GroundTruthPackage> &gt_pack_list){

    // Create a Fusion EKF instance
    FusionEKF fusionEKF;

    // Used to compute the RMSE later
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truth;
    Tools tools;     // compute the accuracy (RMSE)

    size_t N = measurement_pack_list.size();
    for (size_t k = 0; k < N; ++k) {
        //Call the EKF-based fusion
        // start filtering from the second frame (the speed is unknown in the first frame)
        fusionEKF.ProcessMeasurement(measurement_pack_list[k]);

        // Update output file
        write_output(out_file_, fusionEKF, measurement_pack_list[k], gt_pack_list[k]);

        // Update estimations and ground truth to compute RMSE
        estimations.push_back(fusionEKF.ekf_.x_);
        ground_truth.push_back(gt_pack_list[k].gt_values_);
    }

    cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

}

void write_output(ofstream &out_file_, const FusionEKF &fusionEKF,  const MeasurementPackage &curr_measurement, const GroundTruthPackage &curr_gt){

    // Output the state estimation
    out_file_ << fusionEKF.ekf_.x_(0) << "\t";
    out_file_ << fusionEKF.ekf_.x_(1) << "\t";
    out_file_ << fusionEKF.ekf_.x_(2) << "\t";
    out_file_ << fusionEKF.ekf_.x_(3) << "\t";

    // Output the measurements
    if (curr_measurement.sensor_type_ == MeasurementPackage::LASER) {
        // output the estimation
        out_file_ << curr_measurement.raw_measurements_(0) << "\t";
        out_file_ << curr_measurement.raw_measurements_(1) << "\t";
    }
    else if (curr_measurement.sensor_type_ == MeasurementPackage::RADAR) {
        // output the estimation in the cartesian coordinates
        float ro = curr_measurement.raw_measurements_(0);
        float phi = curr_measurement.raw_measurements_(1);
        out_file_ << ro * cos(phi) << "\t"; // p1_meas
        out_file_ << ro * sin(phi) << "\t"; // ps_meas
    }

    // Output the ground truth packages
    out_file_ << curr_gt.gt_values_(0) << "\t";
    out_file_ << curr_gt.gt_values_(1) << "\t";
    out_file_ << curr_gt.gt_values_(2) << "\t";
    out_file_ << curr_gt.gt_values_(3) << "\n";
};