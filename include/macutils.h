//
// Created by Alexander Liu on 8/26/26.
//

#ifndef FLUIDSIM_MACUTILS_H
#define FLUIDSIM_MACUTILS_H

// included by solver
#include "mac.h"

enum class MACIndex : int {
    U = 0,
    V = 1,
};

namespace mac_utils {

}

// numerical interpolation
namespace mac_utils::interpolate {
    // bilinear
    template<std::floating_point T>
    struct Bilinear {
        T operator()(const std::array<T, 2>& pos, const MAC<T, 2>& mac, MACIndex mac_idx = MACIndex::U) {
            // do we need to interpolate the corner points first? 45
            // I think so, we average the 4 points, as shown at end of chapter 2 (46)
            // 2.13.5
            // unsure if need to address edge case of being exactly on grid line
            auto [x,y] = pos;
            int x1 = std::floor(x);
            int x2 = x1+1;
            int y1 = std::floor(y);
            int y2 = y+1;
            // fetch grid corner points
            auto [u_top_left, u_bottom_left, u_top_right, u_bottom_right] = mac.get_cell_velocities(static_cast<int>(pos[0]), static_cast<int>(pos[1]));
            // can prob do smth better than this

            // can optimize recalculations
            T fxy1 = (x2-x)/(x2-x1) * u_bottom_left[mac_idx] + (x-x1)/(x2-x1) * u_bottom_right[mac_idx];

            T fxy2 = (x2-x)/(x2-x1) * u_top_left[mac_idx] + (x-x1)/(x2-x1) * u_top_right[mac_idx];

            T fxy = (y2-y)/(y2-y1) * fxy1[mac_idx] + (y-y1)/(y2-y1) * fxy2[mac_idx];

            return fxy;
        }
    };

    // catmull rom
}

// numerical integrators
namespace mac_utils::integrate {
    // maybe build a functor factory
    // going to do minimum rk3 for impl
    // forward euler
    // rk2
    // rk3
    template<std::floating_point T, int DIM>
    requires(DIM == 2 || DIM == 3)
    struct RK3{};

    template<std::floating_point T>
    struct RK3<T,2> {
        std::array<T,2> operator()(const std::array<T, 2>& pos, const std::array<T,2>& qn, const MAC<T, 2>& mac, double dt, const std::vector<std::vector<Label>>& labels) {
            /*
             * k1 = f(q^n)
             * k2 = f(q^n + 1/2 * dt k1)
             * k3 = f(q^n + 3/4 dt k2)
             * q^{n+1} = q^n + 2/9 dt k1 + 3/9 dt k2 + 4/9 dt k3
             */
            auto interp_f = mac_utils::interpolate::Bilinear<T>();
            std::array<T, 2> k1 = interp_f(pos, mac);
            std::array<T,2> tmp_pos = {
                    pos[0] + 0.5 * dt * k1[0],
                    pos[1] + 0.5 * dt * k1[1]
            };
            std::array<T,2> k2 = interp_f(tmp_pos, mac);
            tmp_pos = {
                    pos[0] + 0.75 * dt * k2[0],
                    pos[1] + 0.75 * dt * k2[1]
            };
            // TODO: RUN A POS CHECK HERE. If the position check is OOB or in a solid, return avg of neighbors instead
            // or extrapolation (chapter 4)
            std::array<T,2> k3 = interp_f(tmp_pos, mac);
            std::array<T,2> q_n1 = {
                    qn[0] + 2./9*dt*k1[0] + 3./9*dt*k2[0] + 4./9*dt*k3[0],
                    qn[1] + 2./9*dt*k1[1] + 3./9*dt*k2[1] + 4./9*dt*k3[1]
            };

            return q_n1;
        }
    };
}

namespace mac_utils::interpolate {
    template<> struct Bilinear<double>;
}

namespace mac_utils::integrate {
    template<> struct RK3<double,2>;
}

#endif //FLUIDSIM_MACUTILS_H
