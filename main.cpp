#include <stdio.h>
#include <vector>

template<int i> struct typeselector {
    typedef typename typeselector<i-1>::type type;
};

template<> struct typeselector<0> { typedef unsigned int type; };
template<> struct typeselector<2> { typedef size_t type; };

template<typename n_t, typename... rest_ts>
class ndoffset {
    typedef ndoffset< rest_ts... > rest_t;
public:
    static const int mults = rest_t::mults + 1;
private:
    typedef typename typeselector<mults>::type offset_t;
    const rest_t rest;
    const n_t n;
public:
    ndoffset(const n_t n_, const rest_ts&... rest_) : n(n_), rest(rest_...) {}
    template<typename i_t, typename... other_ts>
    offset_t calc(i_t i, const other_ts&... other) const {
        return rest.calc(other...) * n + i;
    }
};

template<typename n_t>
class ndoffset<n_t> {
    const n_t n;
public:
    static const int mults = 0;
    ndoffset(const n_t n_) : n(n_) {}
    template<typename i_t>
    n_t calc(i_t i) const {
        return i;
    }
};

template<typename T, typename... n_ts> class ndview {
    typedef ndoffset< n_ts... > offset_t;
    const offset_t offset;
    T* const ptr;
public:
    ndview(T* const ptr_, const n_ts&... n_) : ptr(ptr_), offset(n_...) {};
    template<typename... i_ts>
    T& at(const i_ts&... i) const {
        auto idx = offset.calc(i...);
        printf("mults: %d, idx:%ld, sizeof(idx):%ld\n", offset_t::mults, (size_t) idx, sizeof(idx));
        return ptr[idx];
    }
};

int main() {
    std::vector<int> vec;
    vec.resize(30*30*30*30);
    ndview x(vec.data(),2,3);

    int val = x.at(1,1);

    printf("value: %d\n",val);
    printf("value: %d\n",ndview(vec.data(), 30).at(1));
    printf("value: %d\n",ndview(vec.data(), 30,30).at(1,1));
    printf("value: %d\n",ndview(vec.data(), 30,30,30).at(1,1,1));
    printf("value: %d\n",ndview(vec.data(), 30,30,30,30).at(1,1,1,1));
}