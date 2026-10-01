//
// Created by Alexander Liu on 8/17/26.
//

// Maybe rename to mac utils, instead of mutils

#ifndef FLUIDSIM_MUTILS_H
#define FLUIDSIM_MUTILS_H

#include <algorithm>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

constexpr double CFL_NUMBER = 5.;

// math utilities
namespace mutils {
    // clamp dt, prob stupid
//    void CFL(double& dt, double dx, double u_max) {
//        dt = std::min(CFL_NUMBER * dx / u_max, dt);
//    }

    template<typename T, typename... Args>
    inline T avg(Args... args) {
        size_t sz = sizeof...(args);
        double sum = (args + ...);
        return sum / static_cast<T>(sz);
    }
}

namespace vec_utils {
    template<typename T, int DIM>
    bool isApproximatelyZero(Eigen::Tensor<T, DIM>& tensor, double eps=1e-9) {
        auto data = tensor.data();
        return !std::any_of(data, data + tensor.size(), [&eps](auto x) {return std::abs(x) <= eps;});
    }

    template<std::floating_point T>
    bool isApproximatelyZero(T val, double eps=1e-9) {
        return std::abs(val) <= eps;
    }

    template<std::floating_point T, int DIM>
    T dot(const Eigen::Tensor<T,DIM>& tensor_a, const Eigen::Tensor<T,DIM>& tensor_b) {
        // should benchmark perf of this vs Tensor::contract
        auto data_a = tensor_a.data();
        T out = std::inner_product(data_a,data_a+tensor_a.size(), tensor_b.data(), 0.);
        return out;
    }

    // https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
    // For Eigen::MatrixXd and Eigen::VectorXd
    template<typename Derived>
    std::string get_shape(const Eigen::EigenBase<Derived>& x) {
        std::ostringstream oss;
        oss << "(" << x.rows() << ", " << x.cols() << ")";
        return oss.str();
    }
}

#endif //FLUIDSIM_MUTILS_H
