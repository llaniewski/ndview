#include "ndview.hpp"
#include <stdio.h>

using namespace ndv;

using size0 = fixed_size<4>;
struct size1tag {};
using size1 = tagged<size1tag, size0>;
struct size2tag {};
using size2 = tagged<size2tag, size0>;
struct size3tag {};
using size3 = global_size<size3tag>;
struct size4tag {};
using size4 = tagged<size4tag, dynamic_size>;

int main() {
    size3::set_size(3);

    size4 s{10};
    // size_tuple<size1, size4, size2> w1;
    // size_tuple<size1, size4> w2{s};

    // ndvector<double, size1, size2> test{};
    // ndvector<double, size4, size1> test{s};
    ndvector<double, size1, size4> test{s};

    for (idx<size1> i; i; ++i) {
        for (idx<size4> j; j < s; ++j) {
            auto off = offset(i, j);
            printf("%d %d %d\n", (int)i.value, (int)j.value, (int)off.value);
        }
    }

    // idx_seq<size4> iset{0, 1, s};
    for (auto i : size_tuple<size4>(s).indexes()) {
        printf("index1: %d\n", (int)i);
    }

    for (auto i : test.sizes.indexes(1, 5)) {
        printf("index2: %d\n", (int)i);
    }

    auto w1 = test.extent<1>();
    auto w2 = test.sizes.subtuple<0, 1>();

    for (auto i : test.indexes<0>()) {
        for (auto j : test.indexes<1>()) {
            printf("index: %d %d %lg\n", (int)i, (int)j, test(i, j));
        }
    }
    for (auto i : test.indexes<0, 1>()) {
        printf("index: %d %lg\n", (int)i, test(i));
    }

    for (idx<size1> i; i; ++i) {
        for (idx<size2> j; j; ++j) {
            auto off = offset(i, j);
            printf("%d %d %d\n", (int)i.value, (int)j.value, (int)off.value);
        }
    }

    ndarray<double, size1, size2> tab_static;
    for (idx<size1> i; i; ++i) {
        for (idx<size2> j; j; ++j) {
            tab_static(i, j) = i.value + j.value;
        }
    }
    for (idx<size1, size2> i; i; ++i) {
        double val = tab_static(i);
        printf("%d: %lg\n", (int)i.value, (double)val);
    }

    ndarray<double, size0, size0> tab_static_agnostic;
    for (idx<size1> i; i; ++i) {
        for (idx<size2> j; j; ++j) {
            tab_static_agnostic(i, j) = i.value + j.value;
        }
    }
    for (idx<size1, size2> i; i; ++i) {
        double val = tab_static_agnostic(i);
        printf("%d: %lg\n", (int)i.value, (double)val);
    }

    ndvector<double, size1, size3> tab_dynamic;
    for (idx<size1> i; i; ++i) {
        for (idx<size3> j; j; ++j) {
            tab_dynamic(i, j) = i.value + j.value;
        }
    }

    ndarray<double, size1, size2>  A;
    ndvector<double, size2, size3> B;
    ndvector<double, size1, size3> C;
    for (idx<size1> i; i; ++i) {
        for (idx<size3> j; j; ++j) {
            C(i, j) = 0;
            for (idx<size2> k; k; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }

    return 0;
}
