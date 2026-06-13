#include <stdio.h>
#include "ndview.hpp"

using namespace ndv;

using tag1 = fixed_size<5>;
using tag2 = fixed_size<4>;
struct tag3tag {};
using tag3 = dynamic_size<tag3tag>;
// template <>
// size_t tag3::size = 0;

int main() {
    tag3::set_size(3);

    for (idx<tag1> i; i; ++i) {
        for (idx<tag2> j; j; ++j) {
            auto off = offset(i,j);
            printf("%d %d %d\n",(int) i.value, (int) j.value, (int) off.value);
        }
    }

    ndarray<double, tag1, tag2> tab_static;
    for (idx<tag1> i; i; ++i) {
        for (idx<tag2> j; j; ++j) {
            tab_static(i,j) = i.value+j.value;
        }
    }
    for (idx<tag1,tag2> i; i; ++i) {
        double val = tab_static(i);
        printf("%d: %lg\n",(int) i.value, (double) val);
    }

    ndvector<double, tag1, tag3> tab_dynamic;
    for (idx<tag1> i; i; ++i) {
        for (idx<tag3> j; j; ++j) {
            tab_dynamic(i,j) = i.value+j.value;
        }
    }


    ndarray<double, tag1, tag2> A;
    ndvector<double, tag2, tag3> B;
    ndvector<double, tag1, tag3> C;
    for (idx<tag1> i; i; ++i) {
        for (idx<tag3> j; j; ++j) {
            C(i,j) = 0;
            for (idx<tag2> k; k; ++k) {
                C(i,j) += A(i,k) * B(k,j);
            }
        }
    }

    return 0;
}
