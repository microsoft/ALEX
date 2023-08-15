// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/* This file contains the classes for linear models and model builders, helpers
 * for the bitmap,
 * cost model weights, statistic accumulators for collecting cost model
 * statistics,
 * and other miscellaneous functions
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <typeinfo>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <bitset>
#include <cassert>
#ifdef _WIN32
#include <intrin.h>
#include <limits.h>
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif
#include "mkl.h"
#include "mkl_lapacke.h"

#ifdef _MSC_VER
#define forceinline __forceinline
#elif defined(__GNUC__)
#define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define forceinline inline __attribute__((__always_inline__))
#else
#define forceinline inline
#endif
#else
#define forceinline inline
#endif

/*** string's limit value ***/
#define STR_VAL_MAX 126
#define STR_VAL_MIN 33

/*** debug print ***/
#define DEBUG_PRINT 0

/*** some utils for multithreading ***/
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define COUT_VAR(this) std::cout << #this << ": " << this << std::endl;
#define UNUSED(var) ((void)var)
#define CACHELINE_SIZE (1 << 6)

namespace alex {

static const size_t desired_training_key_n_ = 10000000; /* desired training key according to Xindex */

struct alignas(CACHELINE_SIZE) RCUStatus;
enum class Result;
struct IndexConfig;

typedef RCUStatus rcu_status_t;
typedef Result result_t;
typedef IndexConfig index_config_t;

/*** MAY BE USED UNDER CIRCUMSTANCES ***/
//extern unsigned int max_key_length;

/* AlexKey class. */
template<class T>
class AlexKey {
 public:
  T *key_arr_ = nullptr;
  unsigned int max_key_length_ = 0;

  AlexKey() {}

  AlexKey(T *key_arr, unsigned int max_key_length)
      : max_key_length_(max_key_length) {
    key_arr_ = new T[max_key_length_];
    std::copy(key_arr, key_arr + max_key_length_, key_arr_);
  }

  AlexKey(unsigned int max_key_length)
      : max_key_length_(max_key_length) {
    key_arr_ = new T[max_key_length_]();
  }

  AlexKey(const AlexKey<T>& other)
      : max_key_length_(other.max_key_length_) {
    key_arr_ = new T[max_key_length_]();
    std::copy(other.key_arr_, other.key_arr_ + max_key_length_, key_arr_);
  }

  ~AlexKey() {
      delete[] key_arr_;
  }

  AlexKey<T>& operator=(const AlexKey<T>& other) {
    if (this != &other) {
      delete[] key_arr_;
      max_key_length_ = other.max_key_length_;
      key_arr_ = new T[other.max_key_length_];
      std::copy(other.key_arr_, other.key_arr_ + other.max_key_length_, key_arr_);
    }
    return *this;
  }

  bool operator< (const AlexKey<T>& other) const {
    assert(max_key_length_ == other.max_key_length_);
    for (unsigned int i = 0; i < max_key_length_; i++) {
      if (key_arr_[i] < other.key_arr_[i]) {return true;}
      else if (key_arr_[i] > other.key_arr_[i]) {return false;}
    }
    return false;
  }

};

/*** Linear model and model builder ***/

// Forward declaration
template <class T>
class LinearModelBuilder;

/* Linear Regression Model */
template <class T>
class LinearModel {
 public:
  double *a_ = nullptr;  // slope, we MUST initialize.
  double b_ = 0.0;  // intercept, we MUST initialize by ourself.
  unsigned int max_key_length_ = 0; 

  LinearModel() = default; 

  LinearModel(double a[], double b, unsigned int max_key_length) : max_key_length_(max_key_length) {
    a_ = new double[max_key_length_];
    for (unsigned int i = 0; i < max_key_length_; i++) {
      a_[i] = a[i];
    }
    b_ = b;
  }

  LinearModel(unsigned int max_key_length) : max_key_length_(max_key_length) {
    a_ = new double[max_key_length_]();
    b_ = 0.0;
  }

  ~LinearModel() {
    delete[] a_;
  }

  explicit LinearModel(const LinearModel& other) : 
    b_(other.b_), max_key_length_(other.max_key_length_) {
      a_ = new double[max_key_length_];
      for (unsigned int i = 0; i < max_key_length_; i++) {
        a_[i] = other.a_[i];
      }
  }

  LinearModel& operator=(const LinearModel& other) {
    if (this != &other) {
      delete[] a_;
      max_key_length_ = other.max_key_length_;
      b_ = other.b_;
      a_ = new double[other.max_key_length_];
      std::copy(other.a_, other.a_ + other.max_key_length_, a_);
    }
    return *this;
  }

  void expand(double expansion_factor) {
    assert(a_ != nullptr);
    for (unsigned int i = 0; i < max_key_length_; i++) {
      a_[i] *= expansion_factor;
    }
    b_ *= expansion_factor;
  }

  inline int predict(AlexKey<T> key) const {
    assert(a_ != nullptr);
    assert (max_key_length_ == key.max_key_length_);
    double result = 0.0;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      result += static_cast<double>(key.key_arr_[i]) * a_[i];
    }
    return static_cast<int>(result + b_);
  }

  inline double predict_double(AlexKey<T> key) const {
    assert(a_ != nullptr);
    assert (max_key_length_ == key.max_key_length_);
    double result = 0.0;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      result += static_cast<double>(key.key_arr_[i]) * a_[i];
    }
    return result + b_;
  }
};

/* LinearModelBuilder acts very similar to XIndex model preparing. */
template<class T>
class LinearModelBuilder {
 public:
  LinearModel<T>* model_;

  LinearModelBuilder(LinearModel<T>* model) : model_(model) {}

  inline void add(const AlexKey<T>& x, double y) {
    assert(model_->max_key_length_ == x.max_key_length_);
    if (model_->max_key_length_ == 1) { //single numeric
      count_++;
      x_sum_ += static_cast<long double>(x.key_arr_[0]);
      y_sum_ += static_cast<long double>(y);
      xx_sum_ += static_cast<long double>(x.key_arr_[0]) * x.key_arr_[0];
      xy_sum_ += static_cast<long double>(x.key_arr_[0]) * y;
      x_min_ = std::min<T>(x.key_arr_[0], x_min_);
      x_max_ = std::max<T>(x.key_arr_[0], x_max_);
      y_min_ = std::min<double>(y, y_min_);
      y_max_ = std::max<double>(y, y_max_);
    }
    else { //string
      training_keys_.push_back(x.key_arr_); //may need to deep copy?
      positions_.push_back(y);
    }
  }

  void build() {
    if (model_->max_key_length_ == 1) { /* single dimension */     
      if (count_ <= 1) {
        model_->a_[0] = 0;
        model_->b_ = static_cast<double>(y_sum_);
        return;
      }

      if (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
        // all values in a bucket have the same key.
        model_->a_[0] = 0;
        model_->b_ = static_cast<double>(y_sum_) / count_;
        return;
      }

      auto slope = static_cast<double>(
        (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
      auto intercept = static_cast<double>(
        (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
      model_->a_[0] = slope;
      model_->b_ = intercept;

      // If floating point precision errors, fit spline
      if (model_->a_[0] <= 0) {
        model_->a_[0] = (y_max_ - y_min_) / (double) (x_max_ - x_min_);
        model_->b_ = -static_cast<double>(x_min_) * model_->a_[0];
      }
      return;
    }

    //from here, it is about string.

    if (positions_.size() <= 1) {
      for (unsigned int i = 0; i < model_->max_key_length_; i++) {
        model_->a_[i] = 0.0;
      }
      if (positions_.size() == 0) {model_->b_ = 0.0;}
      else {model_->b_ = positions_[0];}

      return;
    }

    // trim down samples to avoid large memory usage
    size_t step = 1;
    if (training_keys_.size() > desired_training_key_n_) {
      step = training_keys_.size() / desired_training_key_n_;
    }

    std::vector<size_t> useful_feat_index_;
    for (size_t feat_i = 0; feat_i < model_->max_key_length_; feat_i++) {
      double first_val = (double) training_keys_[0][feat_i];
      for (size_t key_i = 0; key_i < training_keys_.size(); key_i += step) {
        if ((double) training_keys_[key_i][feat_i] != first_val) {
          useful_feat_index_.push_back(feat_i);
          break;
        }
      }
    }

    if (training_keys_.size() != 1 && useful_feat_index_.size() == 0) {
      //std::cout<<"all feats are the same"<<std::endl;
    }
    size_t useful_feat_n_ = useful_feat_index_.size();
    bool use_bias_ = true;

    // we may need multiple runs to avoid "not full rank" error
    int fitting_res = -1;
    while (fitting_res != 0) {
      // use LAPACK to solve least square problem, i.e., to minimize ||b-Ax||_2
      // where b is the actual positions, A is inputmodel_keys
      int m = training_keys_.size() / step;                     // number of samples
      int n = use_bias_ ? useful_feat_n_ + 1 : useful_feat_n_;  // number of features
      double *A = (double *) malloc(m * n * sizeof(double));
      double *b = (double *) malloc(std::max(m, n) * sizeof(double));
      if (A == nullptr || b == nullptr) {
        std::cout<<"cannot allocate memory for matrix A or b\n";
        std::cout<<"at "<<__FILE__<<":"<<__LINE__<<std::endl;
        abort();
      }

      for (int sample_i = 0; sample_i < m; ++sample_i) {
        // we only fit with useful features
        for (size_t useful_feat_i = 0; useful_feat_i < useful_feat_n_;
            useful_feat_i++) {
          A[sample_i * n + useful_feat_i] =
              training_keys_[sample_i * step][useful_feat_index_[useful_feat_i]];
        }
        if (use_bias_) {
          A[sample_i * n + useful_feat_n_] = 1;  // the extra 1
        }
        b[sample_i] = positions_[sample_i * step];
        assert(sample_i * step < training_keys_.size());
      }

      // fill the rest of b when m < n, otherwise nan value will cause failure
      for (int i = m; i < n; i++) {
        b[i] = 0;
      }

      fitting_res = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, 1 /* nrhs */, A,
                                  n /* lda */, b, 1 /* ldb, i.e. nrhs */);

      if (fitting_res > 0) {
        // now we need to remove one column in matrix a
        // note that fitting_res indexes starting with 1
        if ((size_t)fitting_res > useful_feat_index_.size()) {
          use_bias_ = false;
        } else {
          size_t feat_i = fitting_res - 1;
          useful_feat_index_.erase(useful_feat_index_.begin() + feat_i);
          useful_feat_n_ = useful_feat_index_.size();
        }

        if (useful_feat_index_.size() == 0 && use_bias_ == false) {
          std::cout<<"impossible! cannot fail when there is only 1 bias column in matrix a\n";
          std::cout<<"at "<<__FILE__<<":"<<__LINE__<<std::endl;
          abort();
        }
      } else if (fitting_res < 0) {
        printf("%i-th parameter had an illegal value\n", -fitting_res);
        exit(-2);
      }

      // set weights to all zero
      for (size_t weight_i = 0; weight_i < model_->max_key_length_; weight_i++) {
        model_->a_[weight_i] = 0;
      }
      // set weights of useful features
      for (size_t useful_feat_i = 0; useful_feat_i < useful_feat_index_.size();
          useful_feat_i++) {
        model_->a_[useful_feat_index_[useful_feat_i]] = b[useful_feat_i];
      }
      // set bias
      if (use_bias_) {
        model_->b_ = b[n - 1];
      }

      free(A);
      free(b);
      mkl_free_buffers();

#if DEBUG_PRINT
      //for debugging
      //alex::coutLock.lock();
      //std::cout << "current a_ (LMB build): ";
      //for (unsigned int i = 0; i < model_->max_key_length_; i++) {
      //  std::cout << model_->a_[i] << " ";
      //}
      //std::cout << ", current b_ (LMB build) :" << model_->b_ << std::endl;
      //alex::coutLock.unlock();
#endif
    }
    assert(fitting_res == 0);
  }

 private:
  int count_ = 0;
  long double x_sum_ = 0;
  long double y_sum_ = 0;
  long double xx_sum_ = 0;
  long double xy_sum_ = 0;
  T x_min_ = std::numeric_limits<T>::max();
  T x_max_ = std::numeric_limits<T>::lowest();
  double y_min_ = std::numeric_limits<double>::max();
  double y_max_ = std::numeric_limits<double>::lowest();
  std::vector<T*> training_keys_;
  std::vector<double> positions_;
};

/** model building for fanout **/
template <class T, class V>
void build_model_fanout(LinearModel<T> *model, V *values, int num_keys, int fanout) {
  LinearModelBuilder<T> model_builder(model);
  for (int i = 0; i < values; i++) {
    model_builder.add(values[i].first, (double) i / (num_keys - 1));
  }
  model_builder.build();
}

/*** Comparison ***/

struct AlexCompare {
  template <class T1, class T2>
  bool operator()(const AlexKey<T1>& x, const AlexKey<T2>& y) const {
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
        "Comparison types must be numeric.");
    assert(x.max_key_length_ == y.max_key_length_);
    auto x_key_ptr_ = x.key_arr_;
    auto y_key_ptr_ = y.key_arr_;
    for (unsigned int i = 0; i < x.max_key_length_; i++) {
      if (x_key_ptr_[i] < y_key_ptr_[i]) {return true;}
      else if (x_key_ptr_[i] > y_key_ptr_[i]) {return false;}
    }
    return false;
  }
};

/*** Helper methods for bitmap ***/

// Extract the rightmost 1 in the binary representation.
// e.g. extract_rightmost_one(010100100) = 000000100
inline uint64_t extract_rightmost_one(uint64_t value) {
  return value & -static_cast<int64_t>(value);
}

// Remove the rightmost 1 in the binary representation.
// e.g. remove_rightmost_one(010100100) = 010100000
inline uint64_t remove_rightmost_one(uint64_t value) {
  return value & (value - 1);
}

// Count the number of 1s in the binary representation.
// e.g. count_ones(010100100) = 3
inline int count_ones(uint64_t value) {
  return static_cast<int>(_mm_popcnt_u64(value));
}

// Get the offset of a bit in a bitmap.
// word_id is the word id of the bit in a bitmap
// bit is the word that contains the bit
inline int get_offset(int word_id, uint64_t bit) {
  return (word_id << 6) + count_ones(bit - 1);
}

/*** Cost model weights ***/

// Intra-node cost weights
constexpr double kExpSearchIterationsWeight = 20;
constexpr double kShiftsWeight = 0.5;

// TraverseToLeaf cost weights
constexpr double kNodeLookupsWeight = 20;
constexpr double kModelSizeWeight = 5e-7;

/*** Stat Accumulators ***/

struct DataNodeStats {
  double num_search_iterations = 0;
  double num_shifts = 0;
};

// Used when stats are computed using a sample
struct SampleDataNodeStats {
  double log2_sample_size = 0;
  double num_search_iterations = 0;
  double log2_num_shifts = 0;
};

// Accumulates stats that are used in the cost model, based on the actual vs
// predicted position of a key
class StatAccumulator {
 public:
  virtual ~StatAccumulator() = default;
  virtual void accumulate(int actual_position, int predicted_position) = 0;
  virtual double get_stat() = 0;
  virtual void reset() = 0;
};

// Mean log error represents the expected number of exponential search
// iterations when doing a lookup
class ExpectedSearchIterationsAccumulator : public StatAccumulator {
 public:
  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);
    count_++;
  }

  double get_stat() override {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  void reset() override {
    cumulative_log_error_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int count_ = 0;
};

// Mean shifts represents the expected number of shifts when doing an insert
class ExpectedShiftsAccumulator : public StatAccumulator {
 public:
  explicit ExpectedShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  // A dense region of n keys will contribute a total number of expected shifts
  // of approximately
  // ((n-1)/2)((n-1)/2 + 1) = n^2/4 - 1/4
  // This is exact for odd n and off by 0.25 for even n.
  // Therefore, we track n^2/4.
  void accumulate(int actual_position, int) override {
    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;
    count_++;
  }

  double get_stat() override {
    if (count_ == 0) return 0;
    // first need to accumulate statistics for current packed region
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};

// Combines ExpectedSearchIterationsAccumulator and ExpectedShiftsAccumulator
class ExpectedIterationsAndShiftsAccumulator : public StatAccumulator {
 public:
  ExpectedIterationsAndShiftsAccumulator() = default;
  explicit ExpectedIterationsAndShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);

    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;

    count_++;
  }

  double get_stat() override {
    assert(false);  // this should not be used
    return 0;
  }

  double get_expected_num_search_iterations() {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  double get_expected_num_shifts() {
    if (count_ == 0) return 0;
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    cumulative_log_error_ = 0;
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};

/*** Miscellaneous helpers ***/

// https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
inline int pow_2_round_up(int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

// https://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
inline int log_2_round_down(int x) {
  int res = 0;
  while (x >>= 1) ++res;
  return res;
}

// https://stackoverflow.com/questions/1666093/cpuid-implementations-in-c
class CPUID {
  uint32_t regs[4];

 public:
  explicit CPUID(unsigned i, unsigned j) {
#ifdef _WIN32
    __cpuidex((int*)regs, (int)i, (int)j);
#else
    asm volatile("cpuid"
                 : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                 : "a"(i), "c"(j));
#endif
  }

  const uint32_t& EAX() const { return regs[0]; }
  const uint32_t& EBX() const { return regs[1]; }
  const uint32_t& ECX() const { return regs[2]; }
  const uint32_t& EDX() const { return regs[3]; }
};

// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
inline bool cpu_supports_bmi() {
  return static_cast<bool>(CPUID(7, 0).EBX() & (1 << 3));
}

/* utils for multithreading
 * Many of this code is copied from Xindex */

inline void memory_fence() { asm volatile("mfence" : : : "memory"); }

/** @brief Compiler fence.
 * Prevents reordering of loads and stores by the compiler. Not intended to
 * synchronize the processor's caches. */
inline void fence() { asm volatile("" : : : "memory"); }

inline uint64_t cmpxchg(uint64_t *object, uint64_t expected,
                               uint64_t desired) {
  asm volatile("lock; cmpxchgq %2,%1"
               : "+a"(expected), "+m"(*object)
               : "r"(desired)
               : "cc");
  fence();
  return expected;
}

inline uint8_t cmpxchgb(uint8_t *object, uint8_t expected,
                               uint8_t desired) {
  asm volatile("lock; cmpxchgb %2,%1"
               : "+a"(expected), "+m"(*object)
               : "r"(desired)
               : "cc");
  fence();
  return expected;
}

template <class val_t>
struct AtomicVal {
  val_t val_;

  // 60 bits for version
  static const uint64_t version_mask = 0x0fffffffffffffff;
  static const uint64_t lock_mask = 0x1000000000000000;

  // lock - removed - is_ptr
  volatile uint64_t status;

  AtomicVal() : status(0) {}
  AtomicVal(val_t val) : val_(val), status(0) {}

  bool locked(uint64_t status) { return status & lock_mask; }
  uint64_t get_version(uint64_t status) { return status & version_mask; }

  void lock() {
    while (true) {
      uint64_t old = status;
      uint64_t expected = old & ~lock_mask;  // expect to be unlocked
      uint64_t desired = old | lock_mask;    // desire to lock
      if (likely(cmpxchg((uint64_t *)&this->status, expected, desired) ==
                 expected)) {
        return;
      }
    }
  }
  void unlock() { status &= ~lock_mask; }
  void incr_version() {
    uint64_t version = get_version(status);
    UNUSED(version);
    status++;
    assert(get_version(status) == version + 1);
  }

  friend std::ostream &operator<<(std::ostream &os, const AtomicVal &leaf) {
    COUT_VAR(leaf.val_);
    COUT_VAR(leaf.locked);
    COUT_VAR(leaf.version);
    return os;
  }

  // semantics: atomically read the value and the `removed` flag
  val_t read() {
    while (true) {
      uint64_t status = this->status;
      memory_fence();
      val_t curr_val = this->val_;
      memory_fence();

      uint64_t current_status = this->status;
      memory_fence();

      if (unlikely(locked(current_status))) {  // check lock
        continue;
      }

      if (likely(get_version(status) ==
                 get_version(current_status))) {  // check version
        return curr_val;
      }
    }
  }

  bool update(const val_t &val) {
    lock();
    bool res;
    this->val_ = val;
    res = true;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return res;
  }

  bool increment() {
    lock();
    bool res;
    this->val_++;
    res = true;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return res;
  }

  bool decrement() {
    lock();
    bool res;
    this->val_--;
    res = true;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return res;
  }

  bool add(val_t cnt) {
    lock();
    bool res;
    this->val_ += cnt;
    res = true;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return res;
  }

  bool subtract(val_t cnt) {
    lock();
    bool res;
    this->val_ -= cnt;
    res = true;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return res;
  }
};

struct RW_lock {
  int32_t read_cnt = 0;
  bool write_pending = false;

  // 63 bits for version
  static const uint64_t version_mask = 0xefffffffffffffff;
  static const uint64_t lock_mask = 0x1000000000000000;

  // lock - removed - is_ptr
  volatile uint64_t status;

  RW_lock() : status(0) {}

  bool locked(uint64_t status) { return status & lock_mask; }
  uint64_t get_version(uint64_t status) { return status & version_mask; }

  void lock() {
    while (true) {
      uint64_t old = status;
      uint64_t expected = old & ~lock_mask;  // expect to be unlocked
      uint64_t desired = old | lock_mask;    // desire to lock
      if (likely(cmpxchg((uint64_t *)&this->status, expected, desired) ==
                 expected)) {
        return;
      }
    }
  }
  void unlock() { status &= ~lock_mask; }
  void incr_version() {
    uint64_t version = get_version(status);
    UNUSED(version);
    status++;
    assert(get_version(status) == version + 1);
  }

  friend std::ostream &operator<<(std::ostream &os, const RW_lock &leaf) {
    COUT_VAR(leaf.read_cnt);
    COUT_VAR(leaf.write_pending);
    COUT_VAR(leaf.status);
    return os;
  }

  bool increment_rd() {
    lock();
    if (write_pending) {
      unlock();
      return false;
    }
    this->read_cnt++;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return true;
  }

  bool decrement_rd() {
    lock();
    this->read_cnt--;
    memory_fence();
    incr_version();
    memory_fence();
    unlock();
    return true;
  }

  // set up write pending and wait until all reads are finished.
  // only one thread should try waiting to write.
  void write_wait() {
    lock();
    this->write_pending = true;
    memory_fence();
    unlock();
    while (true) {
      uint64_t status = this->status;
      memory_fence();

      int32_t curr_rd_cnt = this->read_cnt;
      memory_fence();

      uint64_t current_status = this->status;
      memory_fence();

      if ((get_version(status) == get_version(current_status))
        && curr_rd_cnt == 0) {  // check version and check if all read finished.
        break;
      }
    }

    return;
  }

  void write_finished() {
    lock();
    this->write_pending = false;
    memory_fence();
    unlock();
  }

};

struct myLock {

  static const uint64_t lock_mask = 0x1000000000000000;

  // lock - removed - is_ptr
  volatile uint64_t status;

  myLock() : status(0) {}

  void lock() {
    while (true) {
      uint64_t old = status;
      uint64_t expected = old & ~lock_mask;  // expect to be unlocked
      uint64_t desired = old | lock_mask;    // desire to lock
      if (likely(cmpxchg((uint64_t *)&this->status, expected, desired) ==
                 expected)) {
        return;
      }
    }
  }
  void unlock() { status &= ~lock_mask; }
};

myLock coutLock;

struct RCUStatus {
  std::atomic<uint64_t> status;
  std::atomic<bool> waiting;
};

enum class Result { ok, failed, retry };

struct IndexConfig {
  double root_error_bound = 32;
  double root_memory_constraint = 1024 * 1024;
  double group_error_bound = 32;
  double group_error_tolerance = 4;
  size_t buffer_size_bound = 256;
  double buffer_size_tolerance = 3;
  size_t buffer_compact_threshold = 8;
  size_t worker_n = 0;
  std::unique_ptr<rcu_status_t[]> rcu_status;
  volatile bool exited = false;
};

index_config_t config;
std::mutex config_mutex;

// TODO replace it with user space RCU (e.g., qsbr)
void rcu_init() {
  config_mutex.lock();
  if (config.rcu_status.get() == nullptr) {abort();}
  for (size_t worker_i = 0; worker_i < config.worker_n; worker_i++) {
    config.rcu_status[worker_i].status = 0;
    config.rcu_status[worker_i].waiting = false;
  }
  config_mutex.unlock();
}

void rcu_alloc() {
  config_mutex.lock();
  if (config.rcu_status.get() == nullptr) {
    config.rcu_status = std::make_unique<rcu_status_t[]>(config.worker_n);
  }
  config_mutex.unlock();
}

void rcu_progress(const uint32_t worker_id) {
  config.rcu_status[worker_id].status++;
}

// wait for all workers
void rcu_barrier() {
  uint64_t prev_status[config.worker_n];
  for (size_t w_i = 0; w_i < config.worker_n; w_i++) {
    prev_status[w_i] = config.rcu_status[w_i].status;
  }
  for (size_t w_i = 0; w_i < config.worker_n; w_i++) {
    while (config.rcu_status[w_i].status <= prev_status[w_i] && !config.exited)
      ;
  }
}

// wait for workers whose 'waiting' is false
void rcu_barrier(const uint32_t worker_id) {
  // set myself to waiting for barrier
#if DEBUG_PRINT
  coutLock.lock();
  std::cout << "t" << worker_id << " - waiting started in rcu_barrier" << std::endl;
  coutLock.unlock();
#endif
  config.rcu_status[worker_id].waiting = true;

  uint64_t prev_status[config.worker_n];
  for (size_t w_i = 0; w_i < config.worker_n; w_i++) {
    prev_status[w_i] = config.rcu_status[w_i].status;
  }
  for (size_t w_i = 0; w_i < config.worker_n; w_i++) {
    // skipped workers that is wating for barrier (include myself)
    while (config.rcu_status[w_i].status <= prev_status[w_i] &&
           !config.rcu_status[w_i].waiting && !config.exited)
      ;
  }
  config.rcu_status[worker_id].waiting = false;  // restore my state
#if DEBUG_PRINT
  coutLock.lock();
  std::cout << "t" << worker_id << " - waiting finished in rcu_barrier" << std::endl;
  coutLock.unlock();
#endif
}

}