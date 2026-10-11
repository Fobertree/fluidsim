//
// Created by Alexander Liu on 8/16/26.
//

#ifndef FLUIDSIM_SOLVER_H
#define FLUIDSIM_SOLVER_H

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>

#include "mac.h"
#include "mutils.h"
#include "macutils.h"

#include "safefp.h"

// SOA over AOS for spatial locality
template<typename T, size_t DIM>
requires(DIM == 2 || DIM == 3)
struct AGrid {};

// 5-point laplacian stencil
// discrete poisson equation
// Adiag(i,j) = A_{(i,j), (i,j)}
// Ax(i,j) = A_{(i,j), (i,j+1)}
// Ay(i,j) = A_{(i,j), (i+1,j)}
template<typename T>
struct AGrid<T, 2> {
public:
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

    void reset() {
        Adiag.setZero();
        Ax.setZero();
        Ay.setZero();
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
    using SCALAR_T = SafeFP<T>;
    explicit Eulerian2D(int i, int j, double dt = 0.5, double rho = 1): lenI(i), lenJ(j), ghostLenI(i), ghostLenJ(j), dt(dt), rho(rho),
                        mac(MAC<T,2>(ghostLenI,ghostLenJ)), a_grid(AGrid<T,2>(ghostLenI,ghostLenJ)) {
        // TODO: add a way to read an env into labels
        // actually (i * j) x 1 but doing 2d for access
        divergence = Eigen::Tensor<T, 2>(ghostLenI,ghostLenJ);
        pressure_unknowns = Eigen::Tensor<T, 2>(ghostLenI,ghostLenJ);
        precon = Eigen::Tensor<T, 2>(i,j);
        //
        render_data = std::vector<float>(lenI * lenJ);

        load_environment();
    }

    [[nodiscard]] Label& label(int i, int j) {
        return mac.label(i,j);
    }
    void iterate();
    const std::vector<float>& get_data();
    [[nodiscard]] const std::array<int, 2> get_dims() const {
        return {lenI, lenJ};
    }
private:
    double dt = 0.5;
    double rho = 1.;
    // should always be dx=1 for now, just keeping as placeholder in case it's wrong
    double dx = 1.;

    int lenI, lenJ;
    // lenI, lenJ + 2 for ghost cells to avoid conditional OOB check spaghetti
    // 1 <= i <= lenI
    int ghostLenI, ghostLenJ;
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
    void load_environment();
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
        int x0 = static_cast<int>(x);
        int x1 = x0+1;
        int y0 = static_cast<int>(y);
        int y1 = y0+1;
        // fetch grid corner points
        auto [u_top_left, u_bottom_left, u_top_right, u_bottom_right] = mac.get_prev_u_velocities(pos);
        // can prob do smth better than this

        // TODO: fix x0=x1 scenarios, guaranteed since mac
        float dx = x1-x0;
        float dy = y1-y0;
        // can optimize recalculations
        T fxy1 = (x1-x)/(dx) * u_bottom_left + (x-x0)/(dx) * u_bottom_right;

        T fxy2 = (x1-x)/(dx) * u_top_left + (x-x0)/(dx) * u_top_right;

        T fxy = (y1-y)/(dy) * fxy1 + (y-y0)/(dy) * fxy2;

        return fxy;
    }

    template<std::floating_point T = double>
    T bilinear_v (const std::array<T, 2>& pos, const MAC<T, 2>& mac, MACIndex mac_idx = MACIndex::U) {
        // do we need to interpolate the corner points first? 45
        // I think so, we average the 4 points, as shown at end of chapter 2 (46)
        // 2.13.5
        // unsure if need to address edge case of being exactly on grid line
        auto [x,y] = pos;
        int x0 = static_cast<int>(x);
        int x1 = x0+1;
        int y0 = static_cast<int>(y);
        int y1 = y0+1;
        // fetch grid corner points
        auto [u_top_left, u_bottom_left, u_top_right, u_bottom_right] = mac.get_prev_v_velocities(pos);
        // can prob do smth better than this

        // TODO: fix x0=x1 scenarios, guaranteed since mac
        float dx = x1-x0;
        float dy = y1-y0;
        // can optimize recalculations
        T fxy1 = (x1-x)/(dx) * u_bottom_left + (x-x0)/(dx) * u_bottom_right;

        T fxy2 = (x1-x)/(dx) * u_top_left + (x-x0)/(dx) * u_top_right;

        T fxy = (y1-y)/(dy) * fxy1 + (y-y0)/(dy) * fxy2;

        return fxy;
    }

    // (backward) numerical integrator
    template <std::floating_point T = double>
    T RK3_u(const std::array<T, 2>& pos, T qn, const MAC<T, 2>& mac, double dt, const std::vector<std::vector<Label>>& labels) {
            /*
             * Ralston's third-order method
             * k1 = f(q^n)
             * k2 = f(q^n + 1/2 * dt k1)
             * k3 = f(q^n + 3/4 dt k2)
             * q^{n+1} = q^n + 2/9 dt k1 + 3/9 dt k2 + 4/9 dt k3
             */
            // strictly 1d integration for now. blinear on 4 points in scalar field
            auto clamp_f = [&mac](const std::array<T,2>& pos_input) {
                std::array<T,2> out;
                out[0] = std::clamp(pos_input[0], 0., mac.getLenI()-1.);
                out[1] = std::clamp(pos_input[1], 0., mac.getLenJ()-1.);
                return out;
            };

            T k1 = bilinear_u<T>(pos, mac);
            std::array<T,2> tmp_pos = clamp_f({pos[0] - 0.5 * dt * k1, 0.5 + pos[1]});

            T k2 = bilinear_u<T>(static_cast<const std::array<double, 2>>(tmp_pos), mac);
            tmp_pos = clamp_f({pos[0] - 0.75 * dt * k2, 0.5 + pos[1]});
            // TODO: RUN A POS CHECK HERE. If the position check is OOB or in a solid, return avg of neighbors instead
            // or extrapolation (chapter 4)
            T k3 = bilinear_u<T>(tmp_pos, mac);
            T q_n1 = qn - dt * (2./9*k1 + 3./9*k2 + 4./9*k3);

            return q_n1;
    };

    template <std::floating_point T = double>
    T RK3_v(const std::array<T, 2>& pos, T qn, const MAC<T, 2>& mac, double dt, const std::vector<std::vector<Label>>& labels) {
        /*
         * Ralston's third-order method
         * k1 = f(q^n)
         * k2 = f(q^n - 1/2 * dt k1)
         * k3 = f(q^n - 3/4 dt k2)
         * q^{n+1} = q^n - dt * (2/9 k1 + 3/9 k2 + 4/9 k3)
         */
        // strictly 1d integration for now. blinear on 4 points in scalar field
        auto clamp_f = [&mac](const std::array<T,2>& pos_input) {
            std::array<T,2> out;
            out[0] = std::clamp(pos_input[0], 0., mac.getLenI()-1.);
            out[1] = std::clamp(pos_input[1], 0., mac.getLenJ()-1.);
            return out;
        };

        T k1 = bilinear_v<T>(pos, mac);
        std::array<T,2> tmp_pos = clamp_f({pos[0] - 0.5, pos[0] + 0.5 * dt * k1});

        T k2 = bilinear_v<T>(tmp_pos, mac);
        tmp_pos = clamp_f({ pos[0] + 0.5, pos[1] - 0.75 * dt * k2 });
        // TODO: RUN A POS CHECK HERE. If the position check is OOB or in a solid, return avg of neighbors instead
        // or extrapolation (chapter 4)
        T k3 = bilinear_v(tmp_pos, mac);
        T q_n1 = qn - dt *  (2./9*k1 + 3./9*k2 + 4./9*k3);

        return q_n1;
    };
}

#endif //FLUIDSIM_SOLVER_H
