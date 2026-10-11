//
// Created by Alexander Liu on 10/3/26.
//

#ifndef FLUIDSIM_SAFEFP_H
#define FLUIDSIM_SAFEFP_H

#include <numeric>

// feenableexcept does not appear to be in MacOS, so cannot trap on FP Errors
// not sure if there's still utility in cfenv, prob not
// Would much rather prefer to trigger automatic trap instruction/exception on FP Error
// This is a much more verbose and higher level way of doing it

template <std::floating_point T>
class SafeFP {
public:
    explicit SafeFP() : val(0) {};
    explicit SafeFP(T value): val(value) {
        validate();
    };
    ~SafeFP() = default;

    SafeFP& operator=(const T& other) {
        val = other;
        validate();
        return *this;
    }

    SafeFP& operator=(const SafeFP<T>& other) {
        if (this == &other) return *this;
        val = other.val;
        validate();
        return *this;
    }

    const SafeFP<T>& get() const {return val;}

    std::ostream& operator<<(std::ostream& os) {
        os << val;
        return os;
    }

    // binary arithmetic (FP)
    SafeFP operator+(const T& o) const {return SafeFP(val + o);}
    SafeFP operator-(const T& o) const {return SafeFP(val - o);}
    SafeFP operator*(const T& o) const {return SafeFP(val * o);}
    SafeFP operator/(const T& o) const {return SafeFP(val / o);}
    // compound assignment (FP)
    SafeFP& operator+=(const T& o) const {return val + o; }
    SafeFP& operator-=(const T& o) const {return val - o; }
    SafeFP& operator*=(const T& o) const {return val * o; }
    SafeFP& operator/=(const T& o) const {return val / o; }

    // binary arithmetic (SafeFP)
    SafeFP operator+(const SafeFP& o) const {return SafeFP(val + o);}
    SafeFP operator-(const SafeFP& o) const {return SafeFP(val - o);}
    SafeFP operator*(const SafeFP& o) const {return SafeFP(val * o);}
    SafeFP operator/(const SafeFP& o) const {return SafeFP(val / o);}

    // compound assignment (SafeFP)
    SafeFP& operator+=(const SafeFP& o) const {return this + o; }
    SafeFP& operator-=(const SafeFP& o) const {return this - o; }
    SafeFP& operator*=(const SafeFP& o) const {return this * o; }
    SafeFP& operator/=(const SafeFP& o) const {return this / o; }
private:
    T val;
    inline void validate() const {
        if (std::isnan(val)) {
            throw std::runtime_error("SafeFP::NAN");
        }

        if (std::isinf(val)) {
            throw std::runtime_error("SafeFP::INF");
        }
    }
};

template<std::floating_point T>
std::ostream& operator<<(std::ostream& os, const SafeFP<T>& fp) {
    os << fp;
    return os;
}

using SafeFloat = SafeFP<float>;
using SafeDouble = SafeFP<double>;

// need to split into two concepts because concept cannot have associated constraints (constrained declarations)
// https://eel.is/c++draft/temp#constr.decl

// set up IsSafeFP without requiring internal explicit template args
template<typename T>
struct is_safe_fp : std::false_type {};

template<std::floating_point T>
struct is_safe_fp<SafeFP<T>> : std::true_type {};

template<typename T>
concept IsSafeFP = is_safe_fp<T>();

template<typename T>
concept FP_Union = std::is_floating_point_v<T> || IsSafeFP<T>;

#ifndef N_DEBUG_SAFE_FP
typedef SafeFloat float_type;
typedef SafeDouble double_type;
#else
typedef float float_type;
typedef double double_type;
#endif

#endif //FLUIDSIM_SAFEFP_H
