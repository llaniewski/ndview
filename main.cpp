#include <stdio.h>
#include <vector>

template<int promote, int i> struct typeselector {
    typedef typename typeselector<promote, i-1>::type type;
};

template<int promote> struct typeselector<promote,-1> { typedef unsigned int type; };
template<int promote> struct typeselector<promote,promote> { typedef size_t type; };

template<int promote, typename n_t, typename... rest_ts>
class ndoffset {
    typedef ndoffset< promote, rest_ts... > rest_t;
public:
    static const int mults = rest_t::mults + 1;
private:
    typedef typename typeselector<promote, mults>::type offset_t;
    const rest_t rest;
    const n_t n;
public:
    ndoffset(const n_t n_, const rest_ts&... rest_) : n(n_), rest(rest_...) {}
    template<typename i_t, typename... other_ts>
    offset_t calc(i_t i, const other_ts&... other) const {
        return static_cast<offset_t>(rest.calc(other...)) * n + i;
    }
};

template<int promote,typename n_t>
class ndoffset<promote, n_t> {
    const n_t n;
public:
    static const int mults = 0;
    ndoffset(const n_t n_) : n(n_) {}
    template<typename i_t>
    n_t calc(i_t i) const {
        return i;
    }
};


template<int promote, typename T, typename... n_ts> class ndview {
    typedef ndoffset< promote, n_ts... > offset_t;
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

template<typename T, typename... n_ts> class ndview2 : public ndview<2, T, n_ts...> {
    typedef ndview<2, T, n_ts...> parent;
public:
    ndview2(T* const ptr_, const n_ts&... n_) : parent(ptr_, n_...) {};
};

int main() {
    std::vector<int> vec;
    vec.resize(30*30*30*30);

    printf("value: %d\n",ndview2(vec.data(), 30).at(1));
    printf("value: %d\n",ndview2(vec.data(), 30,30).at(1,1));
    printf("value: %d\n",ndview2(vec.data(), 30,30,30).at(1,1,1));
    printf("value: %d\n",ndview2(vec.data(), 30,30,30,30).at(1,1,1,1));
}