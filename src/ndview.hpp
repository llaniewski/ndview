#ifndef NDVIEW_HPP
#define NDVIEW_HPP

#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <array>

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

    #ifdef __CUDACC__
        template <typename T>
        struct gpu_dynamic_size {
            static inline size_t size_;
            static void set_size(size_t size__) {
                size_ = size__;
                check_cuda(cudaMemcpyToSymbol(T::gpu_size(), &size_, sizeof(size_)),"cudaMemcpyToSymbol(B_size)");
            }
            gpuHD static size_t size() {
                #ifdef __CUDA_ARCH__
                    return T::gpu_size();
                #else
                    return size_;
                #endif
            }
        };

        #define GPU_SIZE(Type) \
            __constant__ size_t Type##_size_gpu; \
            struct Type##_size_t { gpuHD static size_t& gpu_size() { return Type##_size_gpu; } }; \
            using Type = gpu_dynamic_size< Type##_size_t >
    #endif


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
        gpuHD idx& operator+=(const size_t& val) { value+=val; return *this; };
        gpuHD const idx& operator*() const { return *this; };
        explicit gpuHD operator bool() const { return value < size(); };
        gpuHD operator size_t() const { return value; };
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

    template <typename T, typename S, typename... Ns>
    class ndview_generic {
    public:
        using storage_t = S;
        using value_t = T;
        using idx_t = idx<Ns...>;
        static constexpr auto size() { return idx_t::size(); }
        S tab;
        // template <typename... T> 
        // constexpr ndview_generic(T... ts) : tab{ts...} {}
        constexpr gpuHD T& at(const idx_t& i) {
            return tab[i.value];
        };
        constexpr gpuHD const T& at(const idx_t& i) const {
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
        gpuHD T*& data() { return this->tab; }
        gpuHD const T*& data() const { return this->tab; }
    };

    template <typename T, typename... Ns>
    class ndarray : public ndview_generic<T, T[idx<Ns...>::size()], Ns...> {
    public:
        using ndv_t = ndview_generic<T, T[idx<Ns...>::size()], Ns...>;
        gpuHD T* data() { return this->tab; }
        gpuHD const T* data() const { return this->tab; }
        constexpr ndarray() : ndv_t{} {}
        template <class T2>
        constexpr ndarray(const std::array<T2,idx<Ns...>::size()>& init) : ndv_t{} {
            size_t s = idx<Ns...>::size();
            // for (size_t i=0;i<s;++i) this->tab[i] = 1;
            for (size_t i=0;i<s;++i) this->tab[i] = init[i];
            // size_t k=0;
            // for (const auto& v : init) {
            //     if (k<ndv_t::size()) this->tab[k] = v;
            //     ++k;
            // }
        }
        constexpr ndarray(const std::initializer_list<T> init): ndv_t{} {
            size_t k=0;
            for (const auto& v : init) {
                if (k<ndv_t::size()) this->tab[k] = v;
                ++k;
            }
        }
    };


    template <typename T, typename... Ns>
    class ndvector : public ndview_generic<T, std::unique_ptr<T[]>, Ns...> {
    public:
        using ndv_t = ndview_generic<T, std::unique_ptr<T[]>, Ns...>;
        ndvector() : ndv_t{std::unique_ptr<T[]>{new T[ndv_t::size()]}} {}
        ndvector(std::initializer_list<T> init) : ndv_t{std::unique_ptr<T[]>{new T[ndv_t::size()]}} {
            if (init.size() != ndv_t::size()) throw std::runtime_error("Initializer list of wrong size in ndvector");
            std::copy(init.begin(), init.end(), this->tab.get()); // Don't know how to do it nicer.
        }
        gpuH T* data() { return this->tab.get(); }
        gpuH const T* data() const { return this->tab.get(); }
    };

}
#endif //NDVIEW_HPP
