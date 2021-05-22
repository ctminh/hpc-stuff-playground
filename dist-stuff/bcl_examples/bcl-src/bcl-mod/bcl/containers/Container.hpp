#pragma once

#include <iostream>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <math.h>
#include <omp.h>

#include <bcl/bcl.hpp>
int alloced = 0;

namespace BCL {
  template <typename T, size_t N>
  struct serial_blob {
    T val[N];
    size_t len;

    void print() {
      for (int i = 0; i < len; i++) {
        std::cout << val[i];
      }
      std::cout << std::endl;
    }
  };

  template <typename T>
  struct serial_ptr {
    std::shared_ptr <T> ptr;
    size_t N;

    using type = T;

    serial_ptr(const T *ptr, const size_t N) : N(N) {
      this->ptr = std::shared_ptr <T> (ptr);
    }

    serial_ptr(const size_t N) : N(N) {
      this->ptr = std::shared_ptr <T> (new T[N]);
    }

    serial_ptr() {}

    void print() {
      for (int i = 0; i < N; i++) {
        std::cout << ptr[i];
      }
      std::cout << std::endl;
    }
  };

  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

  template <typename>
  struct is_serial_ptr : std::false_type {};

  template <typename T>
  struct is_serial_ptr <serial_ptr <T>> : std::true_type {};

  template <typename T, size_t N = 0, typename Enabler = void>
  struct serialize;

  template <typename T>
  struct identity_serialize {
    T operator()(const T &val) const noexcept {
      return val;
    }
    T deserialize(const T &val) const noexcept {
      return val;
    }
  };

  template <typename T, typename TSerialize>
  using identity_serializer = std::is_base_of<identity_serialize<T>, TSerialize>;

  template <>
  struct serialize <void> {
    void operator()() const noexcept {}
    void deserialize() const noexcept {}
  };

  template <>
  struct serialize <std::string> {
    serial_ptr <char> operator()(const std::string &string) const noexcept {
      serial_ptr <char> ptr;
      ptr.N = string.length();
      ptr.ptr = std::shared_ptr <char> (new char[ptr.N]);

      for (int i = 0; i < string.length(); i++) {
        ptr.ptr.get()[i] = string[i];
      }

      return ptr;
    }
    std::string deserialize(const serial_ptr <char> &ptr) const noexcept {
     return std::string(ptr.ptr.get(), ptr.N);
    }
  };

  template <>
  struct serialize <task> {
    serial_ptr <double> operator()(const task &task) const noexcept {
      
      serial_ptr <double> ptr;
      ptr.N = task.matrixSize*task.matrixSize*3 + 1;
      ptr.ptr = std::shared_ptr <double> (new double[ptr.N]);

      unsigned long tmp = task.matrixSize;
      tmp <<= 32;
      tmp += task.taskId;
      ptr.ptr.get()[ptr.N-1] = tmp;


      for (int i = 0; i < ptr.N-1; i++) {

        if(i >= (ptr.N-1)*2/3){
          ptr.ptr.get()[i] = task.result[i-(ptr.N-1)*2/3];
        }
        else if(i >= (ptr.N-1)/3){
          ptr.ptr.get()[i] = task.matrix2[i-(ptr.N-1)/3];
        }
        else{
          ptr.ptr.get()[i] = task.matrix[i];
        }
      }
      // printf("[%ld]Freed %d\n", BCL::rank(), task.matrixSize*task.matrixSize*2);
      
      // alloced-=2;
      // if(BCL::rank()==0)
      //   printf("[%ld](s)Alloced container: %d\n", BCL::rank(),alloced);
      
      return ptr;
    }
    task deserialize(const serial_ptr <double> &ptr) const noexcept {
      task t;
      double* matrices = ptr.ptr.get();
      long size = matrices[ptr.N-1];
      size >>= 32;
      t.matrixSize = size;
      t.taskId = (unsigned long)matrices[ptr.N-1] % (size << 32);
      // printf("[%ld]Alloced %d\n", BCL::rank(), t.matrixSize*t.matrixSize*2);
      
      for(int i = 0; i<t.matrixSize*t.matrixSize; i++){
        t.matrix[i] = matrices[i];
        // printf("Matrix[%d]: %lf\n",i,t.matrix[i]);
      }
      for(int i = 0; i<t.matrixSize*t.matrixSize; i++){
        t.matrix2[i] = matrices[i+t.matrixSize*t.matrixSize];
      }
      for(int i = 0; i<t.matrixSize*t.matrixSize; i++){
        t.result[i] = matrices[i+2*t.matrixSize*t.matrixSize];
      }
      // alloced+=2;
      
      // if(BCL::rank()==0)
      //   printf("[%ld](des)Alloced container: %d\n", BCL::rank(),alloced);

      return t;
    }
  };

  // TODO: this is not really what we want; would be prefer
  //       std::is_trivially_copyable<T>, but missing from
  //       icc.
  template <typename T>
  struct serialize <T, 0, BCL::enable_if_t<std::is_trivial<T>::value>> :
    public identity_serialize<T>{};

  template <typename T>
  struct serialize <BCL::GlobalPtr <T>>
    : public identity_serialize <BCL::GlobalPtr <T>> {};

  template <size_t N>
  struct serialize <std::string, N, void> {
    serial_blob <char, N> operator()(const std::string &string) const noexcept {
      serial_blob <char, N> blob;

      for (int i = 0; i < string.length(); i++) {
        blob.val[i] = string[i];
      }
      blob.len = string.length();
      return blob;
    }
    std::string deserialize(const serial_blob <char, N> &blob) const noexcept {
      return std::string(blob.val, blob.len);
    }
  };



  template <typename T, typename Serialize, typename Enabler = void>
  class Container {
  public:
    using serialized_type = typename std::result_of<Serialize(T)>::type;

    serialized_type val;

    constexpr static bool has_serial_ptr = false;

    Container(const T &val, const uint64_t rank = BCL::rank()) {
      set(val, rank);
    }

    Container (const Container &container) {
      val = container.val;
    }

    Container() {};

    void free() {}

    T get() const {
      return Serialize{}.deserialize(val);
    }

    void set(const T &val, uint64_t rank = BCL::rank()) {
      this->val = Serialize{}(val);
    }
  };

  template <typename T, typename Serialize>
  class Container <T, Serialize,
     BCL::enable_if_t <is_serial_ptr<typename std::result_of <Serialize(T)>::type>::value>> {
  public:
    using TS = typename std::result_of <Serialize(T)>::type;
    using SPT = typename TS::type;

    using serialized_type = typename std::result_of <Serialize(T)>::type;

    BCL::GlobalPtr <SPT> ptr = nullptr;
    size_t len = 0;
    constexpr static bool has_serial_ptr = true;

    Container(const T &val, const uint64_t rank = BCL::rank()) {
      set(val, rank);
    }

    Container(const Container &container) {
      ptr = container.ptr;
      len = container.len;
    }

    Container() {};

    void free() {
      // TODO: memory leak.
      if (this->ptr != nullptr && this->ptr.is_local()) {
        BCL::dealloc(this->ptr);
      }
    }

    T get() const {
      if (ptr != nullptr) {
        serial_ptr <SPT> local(len);
        BCL::rget(this->ptr, local.ptr.get(), len);
        return Serialize{}.deserialize(local);
      } else {
        return T();
      }
    }

    void set(const T &val, uint64_t rank = BCL::rank()) {
      // TODO: memory leak.
      if (this->ptr != nullptr && this->ptr.is_local()) {
        BCL::dealloc(this->ptr);
      }
      serial_ptr <SPT> ptr = Serialize{}(val);
      this->ptr = BCL::alloc <SPT> (ptr.N);
      if (this->ptr == nullptr) {
        throw std::runtime_error("BCL Container: ran out of memory");
      }
      this->len = ptr.N;
      BCL::rput(ptr.ptr.get(), this->ptr, ptr.N);
    }
  };

  template <>
  class Container <void, BCL::serialize<void>> {
  public:
    Container() {};
    Container(const Container &container) {}

    void free() {}
    void get() const {}
    void set() {}
  };

  // template <>
  // struct serialize <task> {
  //   serial_ptr  <Container<double, serialize<double>>> operator()(const task &task) const noexcept {
  //     serial_ptr <Container<double, serialize<double>>> ptr;
  //     ptr.N = task.matrixSize*task.matrixSize*2 + 1;
  //     ptr.ptr = std::shared_ptr <Container<double, serialize<double>>> (new Container<double, serialize<double>>[ptr.N]);

  //     ptr.ptr.get()[ptr.N-1] = Container<double, serialize<double>> (task.matrixSize);

  //     for (int i = 0; i < ptr.N/2; i++) {
  //       ptr.ptr.get()[i] = Container<double, serialize<double>> (task.matrix[i]);
  //     }
  //     for(int i = ptr.N/2; i< ptr.N-1; i++){
  //       ptr.ptr.get()[i] = Container<double, serialize<double>> (task.matrix2[i-ptr.N/2]);
  //     }
  //     // printf("[%ld]Freed %d\n", BCL::rank(), task.matrixSize*task.matrixSize*2);
  //     free(task.matrix);
  //     free(task.matrix2);
      
  //     alloced-=2;
  //     // if(BCL::rank()==0)
  //     //   printf("[%ld](s)Alloced container: %d\n", BCL::rank(),alloced);
      
  //     return ptr;
  //   }
  //   task deserialize(const serial_ptr <Container<double, serialize<double>>> &ptr) const noexcept {

  //     task t;
  //     t.matrixSize = ptr.ptr.get()[ptr.N-1].get();
  //     t.matrix = (double*)malloc(sizeof(double) * t.matrixSize * t.matrixSize);
  //     t.matrix2 = (double*)malloc(sizeof(double) * t.matrixSize * t.matrixSize);
  //     // printf("[%ld]Alloced %d\n", BCL::rank(), t.matrixSize*t.matrixSize*2);
      
  //     for(int i = 0; i<t.matrixSize*t.matrixSize; i++){
  //       t.matrix[i] = ptr.ptr.get()[i].get();
  //       // printf("Matrix[%d]: %lf\n",i,t.matrix[i]);
  //     }
  //     for(int i = 0; i<t.matrixSize*t.matrixSize; i++){
  //       t.matrix2[i] = ptr.ptr.get()[i+t.matrixSize*t.matrixSize].get();
  //     }
  //     alloced+=2;
      
  //     // if(BCL::rank()==0)
  //     //   printf("[%ld](des)Alloced container: %d\n", BCL::rank(),alloced);
  //     return t;
  //   }
  // };

  template <typename T>
  struct serialize <std::vector <T>> {
    serial_ptr <Container <T, serialize <T>>> operator()(const std::vector <T> &val) const noexcept {
      serial_ptr <Container <T, serialize <T>>> ptr;
      ptr.N = val.size();
      ptr.ptr = std::shared_ptr <Container <T, serialize <T>>> (new Container <T, serialize <T>> [ptr.N]);

      for (size_t i = 0; i < val.size(); i++) {
        ptr.ptr.get()[i] = Container <T, serialize <T>> (val[i]);
      }

      return ptr;

    }
    std::vector <T> deserialize(const serial_ptr <Container <T, serialize <T>>> &ptr) const noexcept {
      std::vector <T> val;

      for (size_t i = 0; i < ptr.N; i++) {
        val.push_back(ptr.ptr.get()[i].get());
      }

      return val;
    }
  };

  template <typename T, typename TSerialize>
  BCL::GlobalPtr <T> decay_container(BCL::GlobalPtr <BCL::Container <T, TSerialize>> ptr) {
    static_assert(std::is_base_of <identity_serialize <T>, TSerialize>::value == true,
      "Cannot decay a container if not identity serializable");
    return reinterpret_pointer_cast <T> (ptr);
  }
}
