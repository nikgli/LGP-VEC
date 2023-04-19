#pragma once

#include <random>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

// #include <iostream> // Use this for debugging operations

using MATRIX = Eigen::Tensor<double, 2>;

/// @brief Random number generator (mainly used for consts) @return Random float between a and b
double randval_mersenne(double a, double b)
{
    std::random_device                     rd;
    std::mt19937                           mt(rd());
    std::uniform_real_distribution<double> dist(a, b);

    return dist(mt);
}

int randval_mersenne(int a, int b)
{
    std::random_device              rd;
    std::mt19937                    mt(rd());
    std::uniform_int_distribution<> dist(a, b);

    return dist(mt);
}

/// @brief Element wise addition between a and b @return New MATRIX
MATRIX VSUMW(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "a: \n" << a << std::endl;
    // std::cout << "b: \n" << b << std::endl;
    auto samples = a.dimension(0);
    auto a_dim   = a.dimension(1);
    auto b_dim   = b.dimension(1);
    if (a_dim != b_dim) {
        MATRIX sum(samples, 1);
        if (a_dim < b_dim) {
            sum.resize(samples, b_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < b_dim; j++) {
                    sum(i, j) = a(i, 0) + b(i, j);
                }
            }
        }
        else {
            sum.resize(samples, a_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    sum(i, j) = a(i, j) + b(i, 0);
                }
            }
        }
        // std::cout << "sum: \n" << sum << std::endl;
        return sum;
    }
    else {
        MATRIX sum = a + b;
        // std::cout << "sum: \n" << sum << std::endl;
        return sum;
    }
}

/// @brief Element wise subtraction between a and b @return New MATRIX
MATRIX V_M(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "a: \n" << a << std::endl;
    // std::cout << "b: \n" << b << std::endl;
    auto samples = a.dimension(0);
    auto a_dim   = a.dimension(1);
    auto b_dim   = b.dimension(1);
    if (a_dim != b_dim) {
        MATRIX sum(samples, 1);
        if (a_dim < b_dim) {
            sum.resize(samples, b_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < b_dim; j++) {
                    sum(i, j) = a(i, 0) - b(i, j);
                }
            }
        }
        else {
            sum.resize(samples, a_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    sum(i, j) = a(i, j) - b(i, 0);
                }
            }
        }
        // std::cout << "sum: \n" << sum << std::endl;
        return sum;
    }
    else {
        MATRIX sum = a - b;
        // std::cout << "sum: \n" << sum << std::endl;
        return sum;
    }
}

/// @brief Element wise multiplication between a and b @return New MATRIX
MATRIX VprW(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "a: \n" << a << std::endl;
    // std::cout << "b: \n" << b << std::endl;
    auto samples = a.dimension(0);
    auto a_dim   = a.dimension(1);
    auto b_dim   = b.dimension(1);
    if (a_dim != b_dim) {
        MATRIX prod(samples, 1);
        if (a_dim < b_dim) {
            prod.resize(samples, b_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < b_dim; j++) {
                    prod(i, j) = a(i, 0) * b(i, j);
                }
            }
        }
        else {
            prod.resize(samples, a_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    prod(i, j) = a(i, j) * b(i, 0);
                }
            }
        }
        // std::cout << "sum: \n" << prod << std::endl;
        return prod;
    }
    else {
        MATRIX prod = a * b;
        // std::cout << "prod: \n" << prod << std::endl;
        return prod;
    }
}

/// @brief Scalar product between a and b
MATRIX VscalprW(const MATRIX& a, const MATRIX& b)
{
    std::array<int, 1> dimensions = {1};
    // std::cout << "a: \n" << a << std::endl;
    // std::cout << "b: \n" << b << std::endl;
    auto samples = a.dimension(0);
    auto a_dim   = a.dimension(1);
    auto b_dim   = b.dimension(1);
    if (a.dimensions().back() != b.dimensions().back()) {
        MATRIX prod(samples, 1);
        if (a_dim < b_dim) {
            for (auto i = 0; i < samples; i++) {
                prod(i, 0) = 0;
                for (auto j = 0; j < b_dim; j++) {
                    prod(i, 0) += a(i, 0) * b(i, j);
                }
            }
        }
        else {
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    prod(i, 0) += a(i, j) * b(i, 0);
                }
            }
        }
        // std::cout << "prod: \n" << prod << std::endl;
        return prod;
    }
    else {
        auto prod = (a * b);
        // std::cout << "prod: \n" << prod << std::endl;
        auto   res       = static_cast<Eigen::Tensor<double, 1>>(prod.sum(dimensions));
        MATRIX scal_prod = res.reshape(Eigen::array<Eigen::Index, 2>({samples, 1}));
        // std::cout << "Scal prod: \n" << scal_prod << std::endl;
        // MATRIX result(samples, dim);
        // for (int i = 0; i < samples; i++) {
        //     for (int j = 0; j < dim; j++) {
        //         result(i, j) = res(i);
        //     }
        // }
        // std::cout << "result: \n" << result << std::endl;

        return scal_prod;
    }
}

/// @brief Element wise protected division between a and b @return New MATRIX
MATRIX VdivW(const MATRIX& a, const MATRIX& b)
{
    /// @warning Division by zero is not allowed
    // std::cout << "a: \n" << a << std::endl;
    // std::cout << "b: \n" << b << std::endl;
    auto samples = a.dimension(0);
    auto a_dim   = a.dimension(1);
    auto b_dim   = b.dimension(1); 
    if (a_dim != b_dim) {
        MATRIX div(samples, 1);
        if (a_dim < b_dim) {
            div.resize(samples, b_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < b_dim; j++) {
                    if (b(i, j) == 0) {
                        div(i, j) = 1000000.0;
                    }
                    else {
                        div(i, j) = a(i, 0) / b(i, j);
                    }
                }
            }
        }
        else {
            div.resize(samples, a_dim);
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    if (b(i, 0) == 0) {
                        div(i, j) = 1000000.0;
                    }
                    else {
                        div(i, j) = a(i, j) / b(i, 0);
                    }
                }
            }
        }
        // std::cout << "div: \n" << div << std::endl;
        return div;
    }
    else {
        auto temp  = static_cast<MATRIX>(a / b);
        auto isinf = temp.isinf();
        // std::cout << isinf << std::endl;
        std::array<int, 1> dimension  = {1};
        auto               isinf_bool = static_cast<Eigen::Tensor<bool, 1>>(isinf.maximum(dimension))(0);
        // std::cout << "\nFart is: \n" << isinf_bool << std::endl;
        if (isinf_bool == 0)
            return temp;
        else {
            for (auto i = 0; i < samples; i++) {
                for (auto j = 0; j < a_dim; j++) {
                    if (b(i, j) == 0) {
                        // Set it to an arbitrary large number
                        temp(i, j) = randval_mersenne(1000000.0, MAXFLOAT);
                    }
                }
            }
            return temp;
        }
    }
}

/// @brief Mean of each row vector of the MATRIX a @return New MATRIX
MATRIX V_mean(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    std::array<int, 1> dimension  = {1};
    auto               temp_const = static_cast<Eigen::Tensor<double, 1>>(a.mean(dimension));
    MATRIX             mean       = temp_const.reshape(Eigen::array<Eigen::Index, 2>({a.dimension(0), 1}));
    // std::cout << "Mean is: \n" << mean << std::endl;
    // auto   samples    = a.dimension(0);
    // auto   dimensions = a.dimension(1);
    // MATRIX result(samples, dimensions);
    // for (int i = 0; i < samples; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         result(i, j) = temp_const(i);
    //     }
    // }

    // std::cout << "Result is: " << result << std::endl;
    return mean;
}

MATRIX V_std(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    std::array<int, 1> dimension = {1};
    auto               samples   = a.dimension(0);
    auto               a_dim     = a.dimension(1);

    if (a_dim == 1) {
        MATRIX std(samples, 1);
        for (auto i = 0; i < samples; i++) {
            std(i, 0) = a(i, 0);
        }
        return std;
    }
    else {
        auto   means = static_cast<Eigen::Tensor<double, 1>>(a.mean(dimension));
        MATRIX std_diff(samples, a_dim);
        for (auto i = 0; i < samples; i++) {
            for (auto j = 0; j < a_dim; j++) {
                std_diff(i, j) = a(i, j) - means(i);
            }
        }
        std_diff       = std_diff.square();
        auto   std_sum = static_cast<Eigen::Tensor<double, 1>>(std_diff.sum(dimension));
        MATRIX std(samples, 1);
        for (auto i = 0; i < samples; i++) {
            std(i, 0) = (std_sum(i) / (double)(a_dim - 1)); // UB here?
        }
        std = std.sqrt();
        // std::cout << "Std is: \n" << std << std::endl;
        return std;
    }
}

/// @brief Returns the sum of each row vector of the MATRIX a @return New MATRIX
MATRIX V_sum(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    std::array<int, 1> dimension  = {1};
    auto               temp_const = static_cast<Eigen::Tensor<double, 1>>(a.sum(dimension));
    MATRIX             sum        = temp_const.reshape(Eigen::array<Eigen::Index, 2>({a.dimension(0), 1}));
    // std::cout << "Sum is: \n" << sum << std::endl;
    // auto   samples    = a.dimension(0);
    // auto   dimensions = a.dimension(1);
    // MATRIX result(samples, dimensions);
    // for (int i = 0; i < samples; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         result(i, j) = temp_const(i);
    //     }
    // }

    // std::cout << "Result is: " << result << std::endl;
    return sum;
}

/// @brief Returns the maximum of each row vector of the MATRIX a @return New MATRIX
MATRIX V_max(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    std::array<int, 1> dimension  = {1};
    auto               temp_const = static_cast<Eigen::Tensor<double, 1>>(a.maximum(dimension));
    MATRIX             max        = temp_const.reshape(Eigen::array<Eigen::Index, 2>({a.dimension(0), 1}));
    // std::cout << "Max is: \n" << max << std::endl;
    // auto   samples    = a.dimension(0);
    // auto   dimensions = a.dimension(1);
    // MATRIX result(samples, dimensions);
    // for (int i = 0; i < samples; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         result(i, j) = temp_const(i);
    //     }
    // }

    // std::cout << "Result is: " << result << std::endl;
    return max;
}

/// @brief Returns the minimum of each row vector of the MATRIX a @return New MATRIX
MATRIX V_min(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    std::array<int, 1> dimension  = {1};
    auto               temp_const = static_cast<Eigen::Tensor<double, 1>>(a.minimum(dimension));
    MATRIX             min        = temp_const.reshape(Eigen::array<Eigen::Index, 2>({a.dimension(0), 1}));
    // std::cout << "Min is: \n" << min << std::endl;
    // auto   samples    = a.dimension(0);
    // auto   dimensions = a.dimension(1);
    // MATRIX result(samples, dimensions);
    // for (int i = 0; i < samples; i++) {
    //     for (int j = 0; j < dimensions; j++) {
    //         result(i, j) = temp_const(i);
    //     }
    // }

    // std::cout << "Result is: " << result << std::endl;
    return min;
}

/// @brief Returns the sin of each element in the MATRIX a @return New MATRIX
MATRIX V_sin(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX sin(a.dimension(0), a.dimension(1));
    for (auto i = 0; i < a.dimension(0); i++) {
        for (auto j = 0; j < a.dimension(1); j++) {
            sin(i, j) = std::sin(a(i, j));
        }
    }
    // std::cout << "Sin is: \n" << sin << std::endl;
    return sin;
}

/// @brief Returns the cos of each element in the MATRIX a @return New MATRIX
MATRIX V_cos(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX cos(a.dimension(0), a.dimension(1));
    for (auto i = 0; i < a.dimension(0); i++) {
        for (auto j = 0; j < a.dimension(1); j++) {
            cos(i, j) = std::cos(a(i, j));
        }
    }
    // std::cout << "Cos is: \n" << cos << std::endl;
    return cos;
}

/// @brief Returns the log of each element in the MATRIX a @return New MATRIX
MATRIX V_log(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX log(a.dimension(0), a.dimension(1));
    for (auto i = 0; i < a.dimension(0); i++) {
        for (auto j = 0; j < a.dimension(1); j++) {
            if (a(i, j) <= 0)
                log(i, j) = 1000000.0;
            else
                log(i, j) = std::log(a(i, j));
        }
    }
    // std::cout << "Log is: \n" << log << std::endl;
    return log;
}

/// @brief Returns the scalar power to the 2 of each row vector in the MATRIX a @return A column vector
MATRIX V_scalpow2(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX pow(a.dimension(0), 1);
    for (auto i = 0; i < a.dimension(0); i++) {
        pow(i, 0) = 0;
        for (auto j = 0; j < a.dimension(1); j++) {
            pow(i, 0) += std::pow(a(i, j), 2);
        }
    }
    // std::cout << "Pow is: \n" << pow << std::endl;
    return pow;
}

/// @brief Returns the scalar power to the 3 of each row vector in the MATRIX a @return A column vector
MATRIX V_scalpow3(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX pow(a.dimension(0), 1);
    for (auto i = 0; i < a.dimension(0); i++) {
        pow(i, 0) = 0;
        for (auto j = 0; j < a.dimension(1); j++) {
            pow(i, 0) += std::pow(a(i, j), 3);
        }
    }
    return pow;
}

/// @brief Element wise power to the 2 of each row vector in the MATRIX a @return MATRIX
MATRIX V_pow2(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX pow(a.dimension(0), a.dimension(1));
    for (auto i = 0; i < a.dimension(0); i++) {
        for (auto j = 0; j < a.dimension(1); j++) {
            pow(i, j) = std::pow(a(i, j), 2);
        }
    }
    return pow;
}

/// @brief Element wise power to the 3 of each row vector in the MATRIX a @return MATRIX
MATRIX V_pow3(const MATRIX& a, const MATRIX& b)
{
    // std::cout << "A is: \n" << a << std::endl;
    MATRIX pow(a.dimension(0), a.dimension(1));
    for (auto i = 0; i < a.dimension(0); i++) {
        for (auto j = 0; j < a.dimension(1); j++) {
            pow(i, j) = std::pow(a(i, j), 3);
        }
    }
    return pow;
}

/// @note Could add more pow functions here