#include <iostream>

#include <fstream>
#include <sstream>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#define TESTING_CASES 100 // Remember to change this if you change the dataset

using DATA   = Eigen::Tensor<double, 3>;
using MATRIX = Eigen::Tensor<double, 2>;

/// @brief Reads the data from the file.
/// FORMAT: ',' for vec values separation. ';' for separation between entities
/// @param run Which benchmark function to use
/// @param training_data Tensor for holding the training data
/// @param training_outputs Matrix holding the training outputs
/// @param training_cases Training samples
/// @param dimensions
/// @param registers Number of registers to use
void read_data(int run, DATA& training_data, MATRIX& training_outputs, int& training_cases, int& dimensions,
    int& output_dimensions, int& registers, DATA& testing_data, MATRIX& testing_outputs, int& testing_cases)
{
    // Automatically determine the number of training cases, dimensions, output dimensions, number
    // of vectors and number of scalars
    // **********************************************************************************************************

    std::string training_filename = "../../datasets/VEC-GP-Benchmark/";
    training_filename += "F_" + std::to_string(run) + "-training" + ".csv";
    std::string testing_filename = "../../datasets/VEC-GP-Benchmark/";
    testing_filename += "F_" + std::to_string(run) + "-testing" + ".csv";

    // Check if file exists
    std::ifstream file(training_filename);
    std::string   line;
    if (!file.is_open()) {
        std::cout << "Training dataset not found. Please check the path" << std::endl;
        std::exit(0);
    }

    int vector_count             = 0;
    int scalar_count             = 0;
    training_cases               = 0;
    int         last_element_dim = 0;
    std::string last_element_type;
    output_dimensions = 0;
    while (std::getline(file, line, '\r')) {
        training_cases++;
        std::stringstream ss(line);
        std::string       element;
        while (std::getline(ss, element, ';')) {
            // Check if element is a vector
            if (element.find(',') != std::string::npos) {
                vector_count++;
                last_element_type = "vector";
                int dim           = 0;
                for (int i = 0; i < element.length(); i++) {
                    if (element[i] == ',')
                        dim++;
                }
                if (vector_count == 1)
                    dimensions = dim + 1;
                else if (dimensions != dim + 1) {
                    std::cout << "All vectors should be of equal dimensionality. Please check the file"
                              << std::endl;
                    std::exit(0);
                }
                last_element_dim = dim + 1;
            }
            // Otherwise, assume it's a scalar
            else {
                scalar_count++;
                last_element_type = "scalar";
            }
        }
    }
    file.close();
    output_dimensions = last_element_type == "vector" ? last_element_dim : 1;

    int vec_counters =
        last_element_type == "vector" ? (vector_count / training_cases) - 1 : (vector_count / training_cases);
    int scal_counters =
        last_element_type == "scalar" ? (scalar_count / training_cases) - 1 : (scalar_count / training_cases);

    // std::cout << "Number of training cases: " << training_cases << std::endl;
    // std::cout << "Number of vectors: " << vec_counters << std::endl;
    // std::cout << "Number of scalars: " << scal_counters << std::endl;
    // std::cout << "Output: " << last_element_type << " of dimension: " << output_dimensions << std::endl;

    // End of counter
    // **********************************************************************************************************
    // Start of training data reading

    testing_cases = TESTING_CASES;
    training_data.resize(training_cases, vec_counters + scal_counters, dimensions);
    training_outputs.resize(training_cases, output_dimensions);

    file.clear();
    file.open(training_filename, std::ios::in);
    line.clear();
    int case_count = -1;
    while (std::getline(file, line, '\r')) {
        case_count++;
        std::stringstream ss(line);
        std::string       element;
        int               variable_count = -1;
        while (std::getline(ss, element, ';')) {
            variable_count++;
            if (variable_count == vec_counters + scal_counters) {
                if (output_dimensions == 1) {
                    training_outputs(case_count, 0) = std::stof(element);
                }
                else {
                    std::stringstream vec_ss(element);
                    std::string       value;
                    int               dim_count = 0;
                    while (std::getline(vec_ss, value, ',')) {
                        training_outputs(case_count, dim_count) = std::stof(value);
                        dim_count++;
                    }
                }
                continue;
            }
            // Check if element is a vector
            if (element.find(',') != std::string::npos) {
                std::stringstream vec_ss(element);
                std::string       value;
                int               dim_count = 0;
                while (std::getline(vec_ss, value, ',')) {
                    training_data(case_count, variable_count, dim_count) = std::stof(value);
                    dim_count++;
                }
            }
            // Otherwise, assume it's a scalar
            else {
                float value = std::stof(element);
                for (int i = 0; i < dimensions; i++)
                    training_data(case_count, variable_count, i) = value;
            }
        }
    }
    file.close();

    // std::cout << "Data read successfully" << std::endl;
    // std::cout << "Training Data Input: \n" << training_data << std::endl;
    // std::cout << "Training Data Output: \n" << training_outputs << std::endl;

    // **********************************************************************************************************
    // Start of testing data reading

    testing_data.resize(testing_cases, vec_counters + scal_counters, dimensions);
    testing_outputs.resize(testing_cases, output_dimensions);

    file.clear();
    file.open(testing_filename, std::ios::in);
    line.clear();
    if (!file.is_open()) {
        std::cout << "Testing dataset not found. Please check the path" << std::endl;
        std::exit(0);
    }
    case_count = -1;
    while (std::getline(file, line, '\r')) {
        case_count++;
        std::stringstream ss(line);
        std::string       element;
        int               variable_count = -1;
        while (std::getline(ss, element, ';')) {
            variable_count++;
            if (variable_count == vec_counters + scal_counters) {
                if (output_dimensions == 1) {
                    testing_outputs(case_count, 0) = std::stof(element);
                }
                else {
                    std::stringstream vec_ss(element);
                    std::string       value;
                    int               dim_count = 0;
                    while (std::getline(vec_ss, value, ',')) {
                        testing_outputs(case_count, dim_count) = std::stof(value);
                        dim_count++;
                    }
                }
                continue;
            }
            // Check if element is a vector
            if (element.find(',') != std::string::npos) {
                std::stringstream vec_ss(element);
                std::string       value;
                int               dim_count = 0;
                while (std::getline(vec_ss, value, ',')) {
                    testing_data(case_count, variable_count, dim_count) = std::stof(value);
                    dim_count++;
                }
            }
            // Otherwise, assume it's a scalar
            else {
                float value = std::stof(element);
                for (int i = 0; i < dimensions; i++)
                    testing_data(case_count, variable_count, i) = value;
            }
        }
    }
    file.close();

    registers = vec_counters + scal_counters;

    // std::cout << "Data read successfully" << std::endl;
    // std::cout << "Testing Data Input: \n" << testing_data << std::endl;
    // std::cout << "Testing Data Output: \n" << testing_outputs << std::endl;
}