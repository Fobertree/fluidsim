//
// Created by Alexander Liu on 8/16/26.
//

#ifndef FLUIDSIM_SOLVER_H
#define FLUIDSIM_SOLVER_H

#include <unsupported/Eigen/CXX11/Tensor>

#include "mac.h"
#include "mutils.h"
#include "macutils.h"

// https://stackoverflow.com/a/13105528
template<typename T, size_t DIM>
struct ndim_vec {
    typedef std::vector<typename ndim_vec<T, DIM-1>::type> type;
};

template<typename T>
struct ndim_vec<T, 0> {
    typedef T type;
};

// SOA over AOS for spatial locality
template<typename T, size_t DIM>
requires(DIM == 2 || DIM == 3)
struct AGrid {};

// 5-point laplacian stencil
// discrete poisson equation
template<typename T>
struct AGrid<T, 2> {
public:
    typedef std::vector<std::vector<T>> VEC_T;
    AGrid(size_t i, size_t j) {
        // A is (ixj by ixj). Each row should have at MOST 5 non-zero (5-point laplacian stencil, 2D) or 7 non-zero values
        // This, combined with being symmetric, makes an incredibly sparse matrix

        // We store 3 numbers per cell on adjacency (note subscript can be flipped bc symmetric)
        // Diagonal: A_{(i,j), (i,j)}
        Adiag = Eigen::Tensor<T,2>(i,j);
        // AplusX: A_{(i,j), (i+1,j)}
        Ax = Eigen::Tensor<T,2>(i,j);
        // AplusY: A_{(i,j), (i,j+1)}
        Ay = Eigen::Tensor<T,2>(i,j);
    }

    Eigen::Tensor<T, 2> Adiag;
    Eigen::Tensor<T, 2> Ax;
    Eigen::Tensor<T, 2> Ay;
};

enum class SOLVER_TYPE {
    Eulerian2D,
    Eulerian3D
};

template<std::floating_point T = double>
class SolverBase {
public:
    virtual void iterate() = 0;
    // float because of shader
    virtual const std::vector<float>& get_data() = 0;
protected:
private:
    // Dirichlet
    virtual void enforceBC() = 0;
};

class Dummy2D : public SolverBase<double>{
// move fluid up and down or smth
};

template<std::floating_point T = double>
class Eulerian2D : public SolverBase<T> {
public:
    explicit Eulerian2D(int i, int j, double dt = 0.5, double rho = 1, double dx = 0.1): lenI(i), lenJ(j), dt(dt), rho(rho), dx(dx), mac(MAC<T,2>(i,j)), a_grid(AGrid<T,2>(i,j)) {
        // TODO: add a way to read an env into labels
        labels = std::vector<std::vector<Label>>(i+1, std::vector<Label>(j+1, Label::UNKNOWN));

        // actually (i * j) x 1 but doing 2d for access
        divergence = Eigen::Tensor<T, 2>(i,j);
        pressure_unknowns = Eigen::Tensor<T, 2>(i,j);
        precon = Eigen::Tensor<T, 2>(i,j);
        render_data = std::vector<float>(lenI * lenJ);
    }

    [[nodiscard]] Label label(int i, int j) const {
        return labels.at(i).at(j);
    }
    void iterate();
    const std::vector<float>& get_data();
    [[nodiscard]] const std::array<int, 2> get_dims() const {
        return {lenI, lenJ};
    }
private:
    double dt = 0.5;
    double rho = 1.;
    double dx = 1.;

    int lenI, lenJ;
    // TODO: port eigen::tensor to eigen::vector
    MAC<T, 2> mac;
    // 5.4
    Eigen::Tensor<T, 2> pressure_unknowns;  // p
    Eigen::Tensor<T, 2> divergence;         // b
    std::vector<std::vector<Label>> labels;
    Eigen::Tensor<T, 2> precon;
    AGrid<T, 2> a_grid;

    std::vector<float> render_data;

    // TODO: sphere/triangular mesh
    void update_labels();
    // 2.2 for scheme
    // semi-lagrangian (backward) advection
    void advect();
    // apply body forces
    void apply_body_forces();
    // project
    void project();
    // Calculate negative divergence rhs (b, in linear system Ap=b)
    void get_negative_divergence();
    void get_A_grid();
    Eigen::Tensor<T,2> apply_A(const Eigen::Tensor<T,2>& pressure_search);
    // PCG is for single-threaded solve. Explore domain decomp for multithreaded solves
    void get_preconditioner();
    Eigen::Tensor<T,2> apply_preconditioner(const Eigen::Tensor<T, 2>& r);
    // A is (i x j, i x j), p is (i x j, 1), b is (i x j, 1)
    void apply_pcg_to_pressure_system();
    void update_pressure();
    void update_velocities();

    // other utils
    void enforceBC();
};

template<std::floating_point T>
class Eulerian3D : public SolverBase<T> {
public:
    void iterate() {

    }
};

template <double> double RK3_u;
template <double> double RK3_v;

template <double> double bilinear_u(const std::array<double,2>& pos, const MAC<double,2>& mac, MACIndex mac_idx);
template <double> double bilinear_v(const std::array<double,2>& pos, const MAC<double,2>& mac, MACIndex mac_idx);

// temporary interp + numerical integration functions that I will refactor later
namespace eulerian_utils {
    // interpolation
    // Note: using naive linear interpolation implicitly introduces a viscosity term via error accumulation (numerical diffusion)
    // The suggestion is to use sharper interpolation (e.g. Catmull Rom & Bicubic interpolation)
    template<std::floating_point T = double>
    T bilinear_u (const std::array<T, 2>& pos, const MAC<T, 2>& mac, MACIndex mac_idx = MACIndex::U) {
        // do we need to interpolate the corner points first? 45
        // I think so, we average the 4 points, as shown at end of chapter 2 (46)
        // 2.13.5
        // unsure if need to address edge case of being exactly on grid line
        auto [x,y] = pos;
        int x1 = std::floor(x);
        int x2 = std::ceil(x);
        int y1 = std::ceil(y);
        int y2 = std::floor(y);
        // fetch grid corner points
        auto [u_top_left, u_bottom_left, u_top_right, u_bottom_right] = mac.get_u_velocities(pos);
        // can prob do smth better than this

        // can optimize recalculations
        T fxy1 = (x2-x)/(x2-x1) * u_bottom_left + (x-x1)/(x2-x1) * u_bottom_right;

        T fxy2 = (x2-x)/(x2-x1) * u_top_left + (x-x1)/(x2-x1) * u_top_right;

        T fxy = (y2-y)/(y2-y1) * fxy1 + (y-y1)/(y2-y1) * fxy2;

        return fxy;
    }

    template<std::floating_point T = double>
    T bilinear_v (const std::array<T, 2>& pos, const MAC<T, 2>& mac, MACIndex mac_idx = MACIndex::U) {
        // do we need to interpolate the corner points first? 45
        // I think so, we average the 4 points, as shown at end of chapter 2 (46)
        // 2.13.5
        // unsure if need to address edge case of being exactly on grid line
        auto [x,y] = pos;
        int x1 = std::floor(x);
        int x2 = std::ceil(x);
        int y1 = std::ceil(y);
        int y2 = std::floor(y);
        // fetch grid corner points
        auto [u_top_left, u_bottom_left, u_top_right, u_bottom_right] = mac.get_v_velocities(pos);
        // can prob do smth better than this

        // can optimize recalculations
        T fxy1 = (x2-x)/(x2-x1) * u_bottom_left + (x-x1)/(x2-x1) * u_bottom_right;

        T fxy2 = (x2-x)/(x2-x1) * u_top_left + (x-x1)/(x2-x1) * u_top_right;

        T fxy = (y2-y)/(y2-y1) * fxy1 + (y-y1)/(y2-y1) * fxy2;

        return fxy;
    }

    // numerical integrator
    template <std::floating_point T = double>
    T RK3_u(const std::array<T, 2>& pos, T qn, const MAC<T, 2>& mac, double dt, const std::vector<std::vector<Label>>& labels) {
            /*
             * k1 = f(q^n)
             * k2 = f(q^n + 1/2 * dt k1)
             * k3 = f(q^n + 3/4 dt k2)
             * q^{n+1} = q^n + 2/9 dt k1 + 3/9 dt k2 + 4/9 dt k3
             */
            // strictly 1d integration for now. blinear on 4 points in scalar field
            T k1 = bilinear_u<T>(pos, mac);
            std::array<T,2> tmp_pos = {pos[0] + 0.5 * dt * k1, 0.5 + pos[1]};

            T k2 = bilinear_u<T>(static_cast<const std::array<double, 2>>(tmp_pos), mac);
            tmp_pos = {pos[0] + 0.75 * dt * k2, 0.5 + pos[1]};
            // TODO: RUN A POS CHECK HERE. If the position check is OOB or in a solid, return avg of neighbors instead
            // or extrapolation (chapter 4)
            T k3 = bilinear_u<T>(tmp_pos, mac);
            T q_n1 = qn + 2./9*dt*k1 + 3./9*dt*k2 + 4./9*dt*k3;

            return q_n1;
    };

    template <std::floating_point T = double>
    T RK3_v(const std::array<T, 2>& pos, T qn, const MAC<T, 2>& mac, double dt, const std::vector<std::vector<Label>>& labels) {
        /*
         * k1 = f(q^n)
         * k2 = f(q^n + 1/2 * dt k1)
         * k3 = f(q^n + 3/4 dt k2)
         * q^{n+1} = q^n + 2/9 dt k1 + 3/9 dt k2 + 4/9 dt k3
         */
        // strictly 1d integration for now. blinear on 4 points in scalar field
        T k1 = bilinear_v<T>(pos, mac);
        std::array<T,2> tmp_pos = {pos[0] + 0.5, pos[0] + 0.5 * dt * k1};

        T k2 = bilinear_v<T>(tmp_pos, mac);
        tmp_pos = {pos[0]+0.5, pos[1] + 0.75 * dt * k2};
        // TODO: RUN A POS CHECK HERE. If the position check is OOB or in a solid, return avg of neighbors instead
        // or extrapolation (chapter 4)
        T k3 = bilinear_v(tmp_pos, mac);
        T q_n1 = qn + 2./9*dt*k1 + 3./9*dt*k2 + 4./9*dt*k3;

        return q_n1;
    };
}


#endif //FLUIDSIM_SOLVER_H
