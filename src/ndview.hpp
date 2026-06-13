#ifndef NDVIEW_HPP
#define NDVIEW_HPP

#include <memory>
#include <initializer_list>
#include <stdexcept>

#ifdef __CUDACC__
#define gpuHD inline __host__ __device__
#define gpuH  inline __host__
#define gpuD  inline __device__
#else
#define gpuHD inline
#define gpuH inline
#define gpuD inline
#endif


namespace ndv {
    
    template <size_t N> struct fixed_size {
        static constexpr size_t size() { return N; }
    };
    template <typename T> class tagged : public T {
        using tag = T;
    };
    template<typename T> struct type_tagger { using type = T; };

    template <typename T>
    class dynamic_size {
        static inline size_t size_;
    public:
        using tag = T;
        gpuH static size_t size() { return size_; }
        gpuH static void set_size(size_t size__) {
            size_ = size__;
        }
    };


    template <typename... Ns>
    constexpr auto total_size() {
        return (Ns::size() * ... * 1);
    }

    template <typename... Ns>
    class idx {
    public:
        using type = size_t;
        static constexpr auto size() { return total_size<Ns...>(); }
        type value = 0;
        gpuHD idx& operator++() { value++; return *this; };
        gpuHD const idx& operator*() const { return *this; };
        gpuHD operator bool() const { return value < size(); };
        bool operator!=(const idx& other) const { return value != other.value; };
    };


    template <typename... A, typename... B>
    gpuHD idx<A..., B...> offset_combine(const idx<A...>& i, const idx<B...>&j) {
        return idx<A..., B...>{i.value+j.value*i.size()};
    }

    template <typename A, typename... B>
    gpuHD auto offset(const A& i, const B&... j) {
        return offset_combine(i, offset(j...));
    }

    template <typename A>
    gpuHD A offset(const A& i) {
        return i;
    }

    template <typename T, typename... Ns>
    class ndview {
    public:
        using idx_t = idx<Ns...>;
        static constexpr auto size() { return idx_t::size(); }
        T* ptr;
        gpuHD T& at(const idx_t& i) {
            return ptr[i.value];
        };
        gpuHD const T& at(const idx_t& i) const {
            return ptr[i.value];
        };
        template <typename... Is>
        gpuHD T& operator()(const Is&... idxs) {
            return at(offset(idxs...));
        };
        template <typename... Is>
        gpuHD const T& operator()(const Is&... idxs) const {
            return at(offset(idxs...));
        };
    };



    template <typename T, typename... Ns>
    class ndarray {
    public:
        using view_t = ndview<T, Ns...>;
        using idx_t = typename view_t::idx_t;
        static constexpr auto size() { return view_t::size(); }
        T tab[size()];
        gpuHD view_t view() { return view_t{tab}; }
        gpuHD operator view_t() { return view(); }
        gpuHD operator const view_t() const { return view(); }
        gpuHD T* data() { return tab; }
        gpuHD const T* data() const { return tab; }
        template <typename... Is>
        gpuHD T& operator()(const Is&... i) {
            return view()(i...);
        };
    };


    template <typename T, typename... Ns>
    class ndvector {
    public:
        using view_t = ndview<T, Ns...>;
        using idx_t = typename view_t::idx_t;
        static constexpr auto size() { return view_t::size(); }
        std::unique_ptr<T[]> tab{new T[size()]};
        ndvector() {}
        ndvector(std::initializer_list<T> init) {
            if (init.size() != size()) throw std::runtime_error("Initializer list of wrong size in ndvector");
            std::copy(init.begin(), init.end(), tab.get()); // Don't know how to do it nicer.
        }
        
        view_t view() { return view_t{tab.get()}; }
        const view_t view() const { return view_t{tab.get()}; }
        operator view_t() { return view(); }
        operator const view_t() const { return view(); }
        T* data() { return tab.get(); }
        const T* data() const { return tab.get(); }
        template <typename... Is>
        T& operator()(const Is&... i) {
            return view()(i...);
        };
    };

}
#endif //NDVIEW_HPP
