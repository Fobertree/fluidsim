#include <iostream>
#include <chrono>
#include <thread>

#include "solver.h"

template<std::floating_point T>
void Eulerian2D<T>::load_environment() {
    // Will add once there's more complex set of IC to invoke in MAC
}

template<std::floating_point T>
void Eulerian2D<T>::update_labels() {
    // classify as solid, air, fluid
    // not sure if pressure + velocity need to be reset, but not doing so for now
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            if (vec_utils::isApproximatelyZero(pressure_unknowns(i,j))) {
                // 0 pressure -> air (Dirichlet for pressure system)
                mac.label(i,j) = Label::EMPTY;
            }
        }
    }
}

// u^{n+J1} = u - dt * 1/rho \nabla p
template<std::floating_point T>
void Eulerian2D<T>::iterate() {
    // enforce BC both before and after updates
    //update_labels();
    advect();
    enforceBC();
    apply_body_forces();
    enforceBC();
    project();
    enforceBC();
    // swap velocity buffers to push current velocity to prev
    mac.swap_buffers();
}

template<std::floating_point T>
const std::vector<float>& Eulerian2D<T>::get_data() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float minF = 99999;
    float maxF = 0;
    int cnt = 0;
    std::cerr << "DOB\b";
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            // TODO: thread pool, openmp, or execution policy this
            T u_component = mutils::avg<T>(mac.u(i,j), mac.u(i+1,j));
            T v_component = mutils::avg<T>(mac.v(i,j), mac.v(i,j+1));
            size_t idx = ((i-1) * lenJ) + j-1;
            // TODO: make sloppy indexing more aesthetic
            render_data[idx] = std::sqrt(u_component*u_component + v_component*v_component);
            //double time = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count(); // NOLINT
            //render_data[(i * lenJ) + j] = std::sin(i + time * 0.05 * j / lenJ); // NOLINT
            if (std::isnan(render_data[idx])) {
                throw std::runtime_error("NAN DETECTED");
            }
            minF = std::min(minF, render_data[(i*lenJ)+j]);
            maxF = std::max(maxF, render_data[(i*lenJ)+j]);
            cnt++;
        }
    }

    std::cout << std::format("min: {}, max: {}, cnt: {}\n", minF, maxF, cnt);

    std::cerr << "BOB";

    return render_data;
}

/* Advection */

/*
 * advect all quantities on the grid
 */

template<std::floating_point T, int DIM>
requires(DIM == 2 || DIM == 3)
struct RK3{};

template<std::floating_point T>
void Eulerian2D<T>::advect() {
    // poll the MAC grid for velocity grid
    auto integrator_f = RK3<T,2>();
    // advection - advect mac quantities

    // need ending pos, q^n (velocity from mac)
    // run numerical integration to get previous position
    // TODO: numerical integrators should return position x_P
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            // run numerical integration once per dimension
            // counterintuitive vs interpolating into vector field and interpolating back
            // but each dimensional scalar field is independent of the others
            auto mac_u_pos = mac.get_u_mac_pos(i,j);
            auto mac_v_pos = mac.get_v_mac_pos(i,j);
            T u_u = eulerian_utils::RK3_u(mac_u_pos, mac.u_prev(i,j), mac, dt, labels);
            T u_v = eulerian_utils::RK3_v(mac_v_pos, mac.v_prev(i,j), mac, dt, labels);
            mac.u(i,j) = u_u;
            mac.v(i,j) = u_v;
        }
    }
}

/* Force Updates */
template<std::floating_point T>
void Eulerian2D<T>::apply_body_forces() {
    // just gravity for now
    // u^B = u^A + dt * g
    constexpr static T GRAVITY = -10;

    Eigen::Tensor<T,2> v = (mac.v + GRAVITY * dt).eval();
    mac.v = v;
}

/* Projection Utilities */

/*
 * Enforce incompressibility
 * Ensure pressure update preserves 0 divergence in velocity field
 */
template<std::floating_point T>
void Eulerian2D<T>::project() {
    get_negative_divergence();
    std::cerr << "Negative Divergence complete\n";
    get_A_grid();
    std::cerr << "A grid complete\n";
    get_preconditioner();
    apply_pcg_to_pressure_system();
    // TODO: swap buffers here
    update_velocities();
}

// Do I need two MACs? One for fluid velocity, one for solid velocity? I don't need central diff for solid velocity, so maybe 2d array is fine
// If I don't represent it directly (I.e. just by label) idk what happens to the space the solids move into
// Should be fine? Ghost pressure

/*
 * Solve for RHS in \vec u^{n+1} = \vec u - \Inc t \frac{1}{\rho}\nabla p
 */
template<std::floating_point T>
void Eulerian2D<T>::get_negative_divergence() {
    double scale = dt / (rho * dx);
    divergence.setZero();
    // 5.3 on u^{n}
    scale = 1./dx;
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            if (label(i, j) == Label::FLUID) {
                divergence(i, j) -= scale * (
                        mac.u(i + 1, j) - mac.u(i, j) +
                        mac.v(i, j + 1) - mac.v(i, j)
                );

                // 5.4 solids
                // u axis
                if (label(i - 1, j) == Label::SOLID) {
                    divergence(i, j) -= scale * (
                            mac.u(i, j) - mac.u_solid(i, j)
                    );
                }

                if (label(i + 1, j) == Label::SOLID) {
                    divergence(i, j) += scale * (
                            mac.u(i + 1, j) - mac.u_solid(i + 1, j)
                    );
                }

                // v axis
                if (label(i, j - 1) == Label::SOLID) {
                    divergence(i, j) -= scale * (
                            mac.v(i, j) - mac.v_solid(i, j)
                    );
                }

                if (label(i, j + 1) == Label::SOLID) {
                    divergence(i, j) += scale * (
                            mac.v(i, j + 1) - mac.v_solid(i, j + 1)
                    );
                }
            }
        }
    }
}

template<> void Eulerian2D<>::get_A_grid() {
    // 5.5
    double scale = dt / (rho*dx*dx);
    a_grid.reset();

    auto& Adiag = a_grid.Adiag;
    auto& Ax = a_grid.Ax;
    auto& Ay = a_grid.Ay;

    static auto idx = [&](int i, int j) {return i*lenJ + j;};

    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            // CHECK LABEL NOT OOB
            // negative x neighbor
            if (label(i-1,j) == Label::FLUID) {
                Adiag(i,j) += scale;
            }
            // positive x neighbor
            if (label(i+1,j) == Label::FLUID) {
                Adiag(i,j) += scale;
                Ax(i,j) -= scale;
            }
            else if (label(i+1,j) == Label::EMPTY) {
                Adiag(i,j) += scale;
            }

            // negative y neighbor
            if (label(i,j-1) == Label::FLUID) {
                Adiag(i,j) += scale;
            }
            // positive y neighbor
            if (label(i,j+1) == Label::FLUID) {
                Adiag(i,j) += scale;
                Ay(i,j) -= scale;
            }
            else if (label(i,j+1) == Label::EMPTY) {
                Adiag(i,j) += scale;
            }
        }
    }
}

template<std::floating_point T>
Eigen::Tensor<T,2> Eulerian2D<T>::apply_A(const Eigen::Tensor<T,2>& pressure_search) {
    // up to 5 values per row (5-point stencil, discrete Poisson). Each (i,j) adds 3 values per column
    // Ap_search=z
    Eigen::Tensor<T,2> out(lenI, lenJ);
    auto& Adiag = a_grid.Adiag;
    auto& Ax = a_grid.Ax;
    auto& Ay = a_grid.Ay;

    size_t A_size = lenI * lenJ;

    // TODO: possibly bad calc
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            // TODO: offset by 1 on all ghost cell
            // Should have up to 5 values: Aplusx, Aminusx, Adiag, Aplusj, Aminusj
            // Adiag(i,j) * p(i,j)
            out(i,j) = Adiag(i,j) * pressure_search(i,j);
            // Ax(i,j-1) * p(i,j-1)
            out(i,j) += Ax(i,j-1) * pressure_search(i,j-1);

            // Ay(i-1,j) * p(i-1,j)
            out(i,j) += Ay(i-1,j) * pressure_search(i-1,j);

            // Ax(i,j+1) * p(i,j+1)
            out(i,j) += Ax(i,j) * pressure_search(i,j+1);

            // Ay(i,j) * p(i+1,j)
            out(i, j) += Ay(i, j) * pressure_search(i + 1, j);
        }
    }
    return out;
}

// maybe shouldn't be void, to semantically guarantee this executes
// https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html
template<std::floating_point T>
void Eulerian2D<T>::get_preconditioner() {
    // 5.7
    static const double tau = 0.97;     // tuning constant
    static const double sigma = 0.25;   // safety constant

    auto& Adiag = a_grid.Adiag;
    auto& Ax = a_grid.Ax;
    auto& Ay = a_grid.Ay;
    T e;
    auto pow2 = [](auto x) {return x*x;};

    precon.setZero();

    // TODO: check if bounds are correct. Is i,j <= or < lenI/lenJ?
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            if (label(i,j) == Label::FLUID) {
                e = Adiag(i,j)
                        - pow2(Ax(i-1,j) * precon(i-1,j))
                        - pow2(Ay(i,j-1) * precon(i,j-1))
                        -tau * (
                                Ax(i-1,j) * (
                                        Ay(i-1,j)
                                        )
                                        * pow2(precon(i-1,j))
                                +Ay(i,j-1) * (
                                        Ax(i,j-1)
                                        )
                                        * pow2(precon(i,j-1))
                                );

                if (e < sigma * Adiag(i,j)) {
                    e = Adiag(i,j);
                }
                precon(i,j) = 1. / (sqrt(e) + 1e-9);
            }
        }
    }
}

template<std::floating_point T>
Eigen::Tensor<T,2> Eulerian2D<T>::apply_preconditioner(const Eigen::Tensor<T, 2>& r) {
    // 5.8, called in 5.6
    // r is residual, which starts as b
    // This iteration of PCG wants to apply preconditioner every iteration
    // Apply MIC(0) preconditioner on residual (z = Mr). Az = r -> MAp = Mr
    // Solve (LL^T)z = r
    Eigen::Tensor<T,2> aux_q(lenI, lenJ);
    aux_q.setZero();
    // pressure unknowns? Seems to be random axu vector (5.6)
    Eigen::Tensor<T, 2> z(lenI, lenJ);
    z.setZero();

    static auto idx = [](int i, int j) {return i*j;};
    auto& Ax = a_grid.Ax;
    auto& Ay = a_grid.Ay;

    // Lq = r
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            if (label(i,j) == Label::FLUID) {
                T t = r(i,j)
                      - Ax(i-1,j) * precon(i-1,j) * aux_q(i-1,j)
                      - Ay(i-1,j) * precon(i-1,j) * aux_q(i-1,j);
                aux_q(i,j) = t * precon(i,j);
            }
        }
    }

    // L^T z = q
    for (int i = lenI-1; i>= 1; i--) {
        for (int j = lenJ-1; j>=1; j--) {
            if (label(i,j) == Label::FLUID) {
                T t = aux_q(i,j)
                      - Ax(i,j) * precon(i,j) * z(i+1,j)
                      - Ay(i,j) * precon(i,j) * z(i,j+1);

                z(i,j) = t * precon(i,j);
            }
        }
    }
    return z;
}

template<std::floating_point T>
void Eulerian2D<T>::apply_pcg_to_pressure_system() {
    // 5.6
    constexpr static int MAX_ITER = 10;
    constexpr static double TOL = 1e-6;
    pressure_unknowns.setZero();
    auto residual = divergence;
    // TODO: improve the r=0 check
    if (vec_utils::isApproximatelyZero(residual)) {
        return;
    }
    auto z = apply_preconditioner(residual);
    // search vector
    auto s = z;

    // eigen.contract for dot product
    // this could be moved to mutils
    auto product_dims = {Eigen::IndexPair<int>(0,0)};
    T sigma = vec_utils::dot(z, residual);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // As = z
        z = apply_A(s);
        T alpha = sigma/vec_utils::dot(z,s);
        Eigen::Tensor<T,2> tmp = (pressure_unknowns + alpha * s).eval();
        pressure_unknowns = tmp;
        residual -= alpha * z;
        // maximum returns zero dimension tensor
        // https://stackoverflow.com/a/74115830
        Eigen::Tensor<double,0> mx = residual.maximum();
        if (mx(0) <= TOL) {
            update_pressure();
            return;
        }
        z = apply_preconditioner(residual);
        T sigma_new = vec_utils::dot(z,residual);
        T beta = sigma_new / sigma;
        s = (z + beta * s).eval();
        sigma = sigma_new;
    }
    std::cerr << "Apply PCG to Pressure System::ITERATION LIMIT EXCEEDED\n";
    update_pressure();
}

template<std::floating_point T>
void Eulerian2D<T>::update_pressure() {
    // reshape
    Eigen::Map<const Eigen::MatrixXd> tensorMap(pressure_unknowns.data(), ghostLenI, ghostLenI);
    mac.p = tensorMap;
}

template<std::floating_point T>
void Eulerian2D<T>::update_velocities() {
    // 5.1, 5.2
    // TODO: cross-check against figure 5.2 for solid handling
    // Need support for ghost pressure in solids
    double scale = dt / (rho * dx);

    // do we need to separately handle i=0, j=0 row/col? Probably need to extrapolate on very boundaries
    // update - 5.2
    for (int i = 1; i <= lenI; i++) {
        for (int j = 1; j <= lenJ; j++) {
            // u update
            if (label(i,j) == Label::FLUID || label(i+1, j) == Label::FLUID) {
                if (label(i,j) == Label::SOLID || label(i+1,j) == Label::SOLID) {
                    mac.u(i+1,j) = mac.u_solid(i+1, j);
                } else {
                    mac.u(i+1, j) -= scale * (mac.p(i+1,j) - mac.p(i,j));
                }
            } else {
                label(i+1,j) = Label::UNKNOWN;
            }
            // v update
            if (label(i,j) == Label::FLUID || label(i,j+1) == Label::FLUID) {
                if (label(i,j) == Label::SOLID || label(i,j+1) == Label::SOLID) {
                    mac.v(i,j+1) = mac.v_solid(i, j+1);
                } else {
                    mac.v(i,j+1) = scale * (mac.p(i,j+1) - mac.p(i,j));
                }
            } else {
                label(i,j+1) = Label::UNKNOWN;
            }
        }
    }
}

template<std::floating_point T>
void Eulerian2D<T>::enforceBC() {
    // homogeneous Dirichlet BC
    // u dim
    for (int j = 0; j <= ghostLenJ; j++) {
        mac.u(0,j) = 0;
        mac.u(lenI, j) = 0;
    }

    // v dim
    for (int i = 0; i < ghostLenI; i++) {
        mac.v(i,0) = 0;
        mac.v(i, lenJ) = 0;
    }
}


template class Eulerian2D<double>;