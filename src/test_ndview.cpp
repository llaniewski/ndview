#include <stdio.h>
#include "ndview.hpp"

using namespace ndv;

using size1 = fixed_size<5>;
using size2 = fixed_size<4>;
struct size3tag {};
using size3 = dynamic_size<size3tag>;
// template <>
// size_t tag3::size = 0;

int main() {
    size3::set_size(3);

    for (idx<size1> i; i; ++i) {
        for (idx<size2> j; j; ++j) {
            auto off = offset(i,j);
            printf("%d %d %d\n",(int) i.value, (int) j.value, (int) off.value);
        }
    }

    ndarray<double, size1, size2> tab_static;
    for (idx<size1> i; i; ++i) {
        for (idx<size2> j; j; ++j) {
            tab_static(i,j) = i.value+j.value;
        }
    }
    for (idx<size1,size2> i; i; ++i) {
        double val = tab_static(i);
        printf("%d: %lg\n",(int) i.value, (double) val);
    }

    ndvector<double, size1, size3> tab_dynamic;
    for (idx<size1> i; i; ++i) {
        for (idx<size3> j; j; ++j) {
            tab_dynamic(i,j) = i.value+j.value;
        }
    }


    ndarray<double, size1, size2> A;
    ndvector<double, size2, size3> B;
    ndvector<double, size1, size3> C;
    for (idx<size1> i; i; ++i) {
        for (idx<size3> j; j; ++j) {
            C(i,j) = 0;
            for (idx<size2> k; k; ++k) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }

    return 0;
}
