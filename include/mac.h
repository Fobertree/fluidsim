//
// Created by Alexander Liu on 8/17/26.
//

/*
 * This only implements Dense MAC so far
 * Not Sparse MAC
 */

#ifndef FLUIDSIM_MAC_H
#define FLUIDSIM_MAC_H

#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include <functional>

#include "mutils.h"

enum class Label {
    FLUID,
    SOLID,
    EMPTY,  // air
    UNKNOWN // not solved for yet
};

// FDM staggered grid
// Dense MAC
// TODO: either flatten 2d vectors into 1d via struct, or use Eigen Tensor
template<typename T, size_t DIM>
requires (DIM == 2 || DIM == 3)
class MAC {};

template<std::floating_point T>
class MAC<T,2> {
public:
    using MATRIX_T = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    MAC(int i, int j) : lenI(i), lenJ(j) {
        // vert axis
        u = TVEC_T(i+1,j);
        // horiz axis
        v = TVEC_T(i,j+1);

        u_prev = TVEC_T(i+1,j);
        v_prev = TVEC_T(i,j+1);

        p = Eigen::MatrixXd(i, j);

        u_solid = TVEC_T(i+1,j);
        v_solid = TVEC_T(i,j+1);

        // tmp until we have solids, or relabel with solids later
        labels = std::vector<Label>(i*j, Label::FLUID);

        u.setZero();
        v.setZero();
        u_prev.setZero();
        v_prev.setZero();
        // don't need solid for now
        // naive - will be optimized against mesh via hash rather than copying val across every cell
        u_solid.setZero();
        v_solid.setZero();
        p.setZero();

        // setup sim IC
        init_mac_values();
        std::cout << std::format("lenI: {}, lenJ: {}\n", lenI, lenJ);
    }

    inline size_t idx(int i, int j) const {return i*lenJ + j;}

    void init_mac_values() {
        // impose IC u(0,x)
        u.setConstant(1.0);
        u_prev.setConstant(1.0);

        // label grid boundaries as solid
        for (int i = 0; i < lenI; i++) {
            labels[idx(i,0)] = Label::SOLID;
            labels[idx(i,lenJ-1)] = Label::SOLID;
        }

        for (int j = 0; j < lenJ; j++) {
            labels[idx(0,j)] = Label::SOLID;
            labels[idx(lenI-1,j)] = Label::SOLID;
        }
    }

    // DEPRECATE
    [[nodiscard]] std::array<Eigen::Vector2<T>, 4> get_cell_velocities(int i, int j) const {
        // 0 <= i < lenI
        // 0 <= j < lenJ
        return {
                u_ij(i,j),              // top left
                u_ij(i+1,j),        // bottom left
                u_ij(i,j+1),        // top right
                u_ij(i+1,j+1)   // bottom right
        };
    }

    [[nodiscard]] std::array<T, 4> get_u_velocities(const std::array<T,2>& pos) const {
        int fi{static_cast<int>(pos[0])};
        int fj{static_cast<int>(pos[1])};
        return {
            u(fi, fj),  // top left
            u(fi+1,fj), // bottom left
            u(fi,fj+1), // top right
            u(fi+1,fj+1)    // bottom right
        };
    }


    [[nodiscard]] std::array<T, 4> get_v_velocities(const std::array<T,2>& pos) const {
        int fi{static_cast<int>(pos[0])};
        int fj{static_cast<int>(pos[1])};
        return {
                v(fi, fj),
                v(fi+1,fj),
                v(fi,fj+1),
                v(fi+1,fj+1)
        };
    }

    [[nodiscard]] std::array<T, 4> get_prev_u_velocities(const std::array<T,2>& pos) const {
        int fi{static_cast<int>(pos[0])};
        int fj{static_cast<int>(pos[1])};
        return {
                u_prev(fi, fj),  // top left
                u_prev(fi+1,fj), // bottom left
                u_prev(fi,fj+1), // top right
                u_prev(fi+1,fj+1)    // bottom right
        };
    }


    [[nodiscard]] std::array<T, 4> get_prev_v_velocities(const std::array<T,2>& pos) const {
        int fi{static_cast<int>(pos[0])};
        int fj{static_cast<int>(pos[1])};
        return {
                v_prev(fi, fj),
                v_prev(fi+1,fj),
                v_prev(fi,fj+1),
                v_prev(fi+1,fj+1)
        };
    }

    // Note: below is multiplied by dx=1
    // get mac position for u cell
    [[nodiscard]] std::array<T,2> get_u_mac_pos(int i,int j) {
        return clamp_pos({static_cast<T>(i), 0.5+j});
    }

    // gety mac position for v cell
    [[nodiscard]] std::array<T,2> get_v_mac_pos(int i, int j) {
        return clamp_pos({i + 0.5, static_cast<T>(j)});
    }

    void swap_buffers() {
        std::swap(u, u_prev);
        std::swap(v,v_prev);
    }

    Label& label(int i, int j) {
        assert((1 <= lenI && j <= lenJ && i >= 0 && j >= 0) && "OUT OF BOUNDS");

        return labels.at(i*lenJ + j);
    }

    const int getLenI() const {return lenI-1;}
    const int getLenJ() const {return lenJ-1;}

    using TVEC_T = Eigen::Tensor<T,2>;

    // direct mac
    TVEC_T u;
    TVEC_T v;

    // swap buffers
    TVEC_T u_prev;
    TVEC_T v_prev;

    // not sure if this memory overhead is necessary
    TVEC_T u_solid;
    TVEC_T v_solid;

    Eigen::MatrixXd p;

    // swap buffer: swap buffer holds previous velocities
    // we need to double buffer: https://courses.grainger.illinois.edu/CS418/sp2023/text/grid-fluids.html

    // We perform advection + projection directly on the MAC - no auxiliary field needed
private:
    int lenI, lenJ;

    std::vector<Label> labels;

    // For advection - store old quantities in case advection yields OOB
    // 37 - averages mac neighbors into grid cell corners
    // this returns u(i,j) at the center of the cell, based on its mac quantities
    // call when original point from numerical integrator in advection x_G is in a solid
    std::array<T,2> clamp_pos(const std::array<T,2>& pos) {
        std::array<T,2> out;
        out[0] = std::clamp(pos[0], 0., lenI-2.);
        out[1] = std::clamp(pos[1], 0., lenJ-2.);
        return out;
    }

    // This samples one grid corner (for bilinear interp)
    [[nodiscard]] Eigen::Vector2<T> u_ij (int i, int j) const {
        // 0 <= i <= lenI
        // 0 <= j <= lenJ
        Eigen::Vector2<T> out;
        out.setZero();
        uint8_t nei_cnt_u{0};
        uint8_t nei_cnt_v{0};
        // left
        // TODO: ghost cells to get rid of awkward conditionals
        if (j > 0) {
            out(0) += u(i,j-1);
            nei_cnt_u++;
        }
        // up
        if (i > 0) {
            out(1) += v(i-1,j);
            nei_cnt_v++;
        }
        // right
        if (j < lenJ) {
            out(0) += u(i,j);
            nei_cnt_u++;
        }
        // down
        if (i < lenI) {
            out(1) += v(i,j);
            nei_cnt_v++;
        }

        // shouldn't even div by 0
        out(0) /= nei_cnt_u;
        out(1) /= nei_cnt_v;
        return out;
    }

    std::array<Eigen::Vector2<T>,4> getGridPoints(int i, int j) {
        return {
            u_ij(i,j),             // top left
            u_ij(i+1,j),        // bottom left
            u_ij(i,j+1),        // top right
            u_ij(i+1,j+1)   // bottom right
        };
    }
};

template<typename T>
class MAC<T, 3> {
    MAC(int i, int j, int k) {

    }

    std::tuple<T, T, T> foo() {

    }
private:
    std::vector<std::vector<std::vector<T>>> q;
    std::vector<std::vector<std::vector<T>>> p;
};

template class MAC<double,2>;


#endif //FLUIDSIM_MAC_H
