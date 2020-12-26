#ifndef MY_MAKE_UNIQUE
#define MY_MAKE_UNIQUE

#include <memory>

template<typename T, typename... Args>
std::unique_ptr<T> my_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif
