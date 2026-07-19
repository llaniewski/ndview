#ifndef NDVIEW_HPP
#define NDVIEW_HPP

#ifdef __cpp_concepts
#include <concepts>
#endif
#include <array>
#include <initializer_list>
#include <memory>
#include <stdexcept>

#ifdef __CUDACC__
#define gpuHD __host__ __device__
#define gpuH __host__
#define gpuD __device__
#else
#define gpuHD
#define gpuH
#define gpuD
#endif

namespace ndv {

template <size_t N>
struct fixed_size {
    static constexpr size_t size() { return N; }
};
template <typename TAG, typename T>
class tagged : public T {
    using tag = TAG;
};

class dynamic_size {
    size_t size_;

public:
    dynamic_size(size_t size__) : size_{size__} {};
    size_t size() const { return size_; };
};

template <typename T>
class global_size {
    static inline size_t size_;

public:
    using tag = T;
    gpuH static size_t size() { return size_; }
    gpuH static void   set_size(size_t size__) { size_ = size__; }
};

#ifdef __CUDACC__
template <typename S, typename T>
struct gpu_constant {
    static inline S value;
    static void     set(const S& value_) {
        value = value_;
        check_cuda(cudaMemcpyToSymbol(T::gpu_value(), &value, sizeof(S)), "cudaMemcpyToSymbol(...)");
    }
    gpuHD static size_t get() {
#ifdef __CUDA_ARCH__
        return T::gpu_value();
#else
        return value;
#endif
    }
};
#define GPU_CONSTANT(type, name)                                                                                       \
    __constant__ type name##_gpu_value;                                                                                \
    struct name##_gpu_constant_t {                                                                                     \
        gpuHD static type& gpu_value() { return name##_gpu_value; }                                                    \
    };                                                                                                                 \
    using name = gpu_global_size<type, name##_gpu_constant_t>

template <typename T>
struct gpu_global_size {
    static inline size_t size_;
    static void          set_size(size_t size__) {
        size_ = size__;
        check_cuda(cudaMemcpyToSymbol(T::gpu_size(), &size_, sizeof(size_)), "cudaMemcpyToSymbol(B_size)");
    }
    gpuHD static size_t size() {
#ifdef __CUDA_ARCH__
        return T::gpu_size();
#else
        return size_;
#endif
    }
};

#define GPU_SIZE(Type)                                                                                                 \
    __constant__ size_t Type##_size_gpu;                                                                               \
    struct Type##_size_t {                                                                                             \
        gpuHD static size_t& gpu_size() { return Type##_size_gpu; }                                                    \
    };                                                                                                                 \
    using Type = gpu_global_size<Type##_size_t>
#endif

template <typename... Ns>
class size_tuple;
template <typename N, typename... Ns>
class size_tuple<N, Ns...> {
    using rest_t = size_tuple<Ns...>;
    N      my;
    rest_t rest;

public:
    template <typename... Ms>
    constexpr size_tuple(const N& val, const Ms&... v) : my{val}, rest{v...} {};
    template <typename... Ms>
    constexpr size_tuple(const Ms&... v) : my{}, rest{v...} {};
    constexpr size_t size() { return my.size() * rest.size(); }
};
template <typename N>
class size_tuple<N> {
    N my;

public:
    constexpr size_tuple(const N& val) : my{val} {};
    constexpr size_tuple() : my{} {};
    constexpr size_t size() { return my.size(); }
};

template <typename... Ns>
constexpr auto total_size() {
    return (Ns::size() * ... * 1);
}

template <typename... Ns>
constexpr auto total_size(const Ns&... x) {
    return (x.size() * ... * 1);
}

template <typename... Ns>
class idx {
public:
    using type = size_t;
    static constexpr auto size() { return total_size<Ns...>(); }
    type                  value = 0;
    gpuHD idx&            operator++() {
        value++;
        return *this;
    };
    gpuHD idx& operator+=(const size_t& val) {
        value += val;
        return *this;
    };
    gpuHD bool less(const idx& other) const { return value < other.value; };
    gpuHD bool less(const Ns&... x) const { return value < total_size(x...); };
    template <typename T>
    gpuHD bool operator<(const T& other) const {
        return less(other);
    };
    gpuHD const idx& operator*() const { return *this; };
    explicit gpuHD   operator bool() const { return value < size(); };
    explicit gpuHD   operator int() const { return value; };
    // gpuHD operator size_t() const { return value; };
    bool operator!=(const idx& other) const { return value != other.value; };
};

template <typename... A, typename... B>
gpuHD idx<A..., B...> offset_combine(const idx<A...>& i, const idx<B...>& j) {
    return idx<A..., B...>{i.value + j.value * i.size()};
}

template <typename A, typename... B>
gpuHD auto offset(const A& i, const B&... j) {
    return offset_combine(i, offset(j...));
}

template <typename A>
gpuHD A offset(const A& i) {
    return i;
}

template <typename A, typename... B>
gpuHD constexpr auto decompose(const idx<A, B...> i) {
    const size_t    size = idx<A>::size();
    const idx<A>    a{i.value % size};
    const idx<B...> b{i.value / size};
    return std::tuple_cat(std::make_tuple(a), decompose(b));
}
template <typename A>
gpuHD constexpr auto decompose(const idx<A> a) {
    return std::make_tuple(a);
}

#ifdef __cpp_concepts
template <typename A, typename B>
struct idx_convertible_impl : std::false_type {};

template <typename... Ns, typename... Ms>
struct idx_convertible_impl<idx<Ns...>, idx<Ms...>> :
    std::bool_constant<sizeof...(Ns) == sizeof...(Ms) && (... && std::convertible_to<Ms, Ns>)> {};

template <typename A, typename B>
concept idx_convertible = idx_convertible_impl<A, B>::value;
#define REQIRE_CONVERTIBLE requires idx_convertible<idx_t, i_t>
#else
#define REQIRE_CONVERTIBLE
#endif

template <typename T, typename S, typename... Ns>
class ndview_generic {
public:
    using storage_t = S;
    using value_t   = T;
    using idx_t     = idx<Ns...>;
    size_tuple<Ns...> sizes;
    S                 tab;

    constexpr auto size() { return sizes.size(); }

    // template <typename... T>
    // constexpr ndview_generic(T... ts) : tab{ts...} {}
    template <class i_t>
    REQIRE_CONVERTIBLE constexpr gpuHD T& at(const i_t& i) {
        return tab[i.value];
    };
    template <class i_t>
    REQIRE_CONVERTIBLE constexpr gpuHD const T& at(const i_t& i) const {
        return tab[i.value];
    };
    template <typename... Is>
    constexpr gpuHD T& operator()(const Is&... idxs) {
        return at(offset(idxs...));
    };
    template <typename... Is>
    constexpr gpuHD const T& operator()(const Is&... idxs) const {
        return at(offset(idxs...));
    };
};

template <typename T, typename... Ns>
class ndview : public ndview_generic<T, T*, Ns...> {
public:
    using ndv_t = ndview_generic<T, T*, Ns...>;
    gpuHD T*&       data() { return this->tab; }
    gpuHD T* const& data() const { return this->tab; }
    template <size_t SIZE>
    constexpr ndview(T (&table)[SIZE]) : ndv_t{} {
        static_assert(SIZE >= idx<Ns...>::size(), "Too small array provided for ndview");
        this->tab = table;
    }
    constexpr ndview(T* table) : ndv_t{} { this->tab = table; }
    template <typename... Ms>
    constexpr ndview(T* table, const Ms&... other) : ndv_t{other...} {
        this->tab = table;
    }
    constexpr ndview() : ndv_t{} {}
};

template <typename T, typename... Ns>
class ndarray : public ndview_generic<T, T[idx<Ns...>::size()], Ns...> {
public:
    using ndv_t = ndview_generic<T, T[idx<Ns...>::size()], Ns...>;
    gpuHD T*       data() { return this->tab; }
    gpuHD const T* data() const { return this->tab; }
    explicit gpuHD operator ndview<T, Ns...>() const { return data(); };
    constexpr ndarray() : ndv_t{} {}
    template <class T2>
    constexpr ndarray(const std::array<T2, idx<Ns...>::size()>& init) : ndv_t{} {
        size_t s = idx<Ns...>::size();
        // for (size_t i=0;i<s;++i) this->tab[i] = 1;
        for (size_t i = 0; i < s; ++i)
            this->tab[i] = init[i];
        // size_t k=0;
        // for (const auto& v : init) {
        //     if (k<ndv_t::size()) this->tab[k] = v;
        //     ++k;
        // }
    }
    constexpr ndarray(const std::initializer_list<T> init) : ndv_t{} {
        size_t k = 0;
        for (const auto& v : init) {
            if (k < ndv_t::size())
                this->tab[k] = v;
            ++k;
        }
    }
};

template <typename T, typename... Ns>
class ndvector : public ndview_generic<T, T*, Ns...> {
public:
    using ndv_t = ndview_generic<T, T*, Ns...>;
    template <typename... Ms>
    ndvector(const Ms&... other) : ndv_t{other...} {
        this->tab = new T[ndv_t::size()];
    }
    ndvector(std::initializer_list<T> init) : ndv_t{new T[ndv_t::size()]} {
        if (init.size() != ndv_t::size())
            throw std::runtime_error("Initializer list of wrong size in ndvector");
        std::copy(init.begin(), init.end(), this->tab); // Don't know how to do it nicer.
    }
    ~ndvector() { delete[] this->tab; }
    gpuHD         operator ndview<T, Ns...>() { return ndview<T, Ns...>{data()}; };
    gpuH T*       data() { return this->tab; }
    gpuH const T* data() const { return this->tab; }
};

} // namespace ndv
#endif // NDVIEW_HPP
