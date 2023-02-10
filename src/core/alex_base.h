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
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
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

/*** type of key ***/
#define STRING (0)
#define INTEGER (1)
#define DOUBLE (2)

namespace alex {

static const size_t desired_training_key_n_ = 10000000; /* desired training key according to Xindex */

/*** MAY BE USED UNDER CIRCUMSTANCES ***/
//extern unsigned int max_key_length;

/* AlexKey class. */
class AlexKey {
 public:
  double *key_arr_ = nullptr;
  unsigned int max_key_length_ = 0;

  AlexKey() {}

  AlexKey(double *key_arr, unsigned int max_key_length)
      : max_key_length_(max_key_length) {
    key_arr_ = new double[max_key_length_];
    std::copy(key_arr, key_arr + max_key_length_, key_arr_);
  }

  AlexKey(unsigned int max_key_length)
      : max_key_length_(max_key_length) {
    key_arr_ = new double[max_key_length_]();
  }

  AlexKey(const AlexKey& other)
      : max_key_length_(other.max_key_length_) {
    key_arr_ = new double[max_key_length_]();
    std::copy(other.key_arr_, other.key_arr_ + max_key_length_, key_arr_);
  }

  ~AlexKey() {
      delete[] key_arr_;
  }

  AlexKey& operator=(const AlexKey& other) {
    if (this != &other) {
      delete[] key_arr_;
      max_key_length_ = other.max_key_length_;
      key_arr_ = new double[other.max_key_length_];
      std::copy(other.key_arr_, other.key_arr_ + other.max_key_length_, key_arr_);
    }
    return *this;
  }

  bool operator< (const AlexKey& other) const {
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
class LinearModelBuilder;

/* Linear Regression Model */
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

  inline int predict(AlexKey key) const {
    assert(a_ != nullptr);
    assert (max_key_length_ == key.max_key_length_);
    double result = 0.0;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      result = key.key_arr_[i] * a_[i];
    }
    return static_cast<int>(result + b_);
  }

  inline double predict_double(AlexKey key) const {
    assert(a_ != nullptr);
    assert (max_key_length_ == key.max_key_length_);
    double result = 0.0;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      result = key.key_arr_[i] * a_[i];
    }
    return result + b_;
  }

  //Helper related to linear model.
  //calculate the smallest double array that results to 'result' predicted by current model.
  //that specific array is stored in the container.
  void find_minimum (double *start, int result, double *container, int key_type) const {
    if (key_type != STRING) {
      container[0] = (result - b_) / a_[0];
    }
    else {
      if (start != nullptr) {
        std::copy(start, start + max_key_length_, container);
      }
      AlexKey tmpkey = AlexKey(container, max_key_length_);
      while (container[0] != 36.0) {
        if (predict(tmpkey) == result) {
          break;
        }
        container[max_key_length_ - 1] += 1.0;
        for (unsigned int i = max_key_length_ - 1; i > 0; i--) {
          if (container [i] == 36.0) {
            container[i] = 0.0;
            container[i-1] += 1.0;
          }
          else {break;}
        }
      }
    }
  }
};

/* LinearModelBuilder acts very similar to XIndex model preparing. */
class LinearModelBuilder {
 public:
  LinearModel* model_;

  LinearModelBuilder (LinearModel* model) : model_(model) {}

  inline void add(const AlexKey& x, int y) {
    assert(model_->max_key_length_ == x.max_key_length_);
    training_keys_.push_back(x.key_arr_);
    positions_.push_back(y);
  }

  void build() {
    if (positions_.size() == 0) {return;}
    if (positions_.size() == 1) {
      for (unsigned int i = 0; i < model_->max_key_length_; i++) {
        model_->a_[i] = 0.0;
      }
      model_->b_ = positions_[0];
      return;
    }

    if (model_->max_key_length_ == 1) { /* single dimension */
      double x_expected = 0, y_expected = 0, xy_expected = 0,
            x_square_expected = 0;
      for (size_t key_i = 0; key_i < positions_.size(); key_i++) {
        double key = training_keys_[key_i][0];
        x_expected += key;
        y_expected += positions_[key_i];
        x_square_expected += key * key;
        xy_expected += key * positions_[key_i];
      }
      x_expected /= positions_.size();
      y_expected /= positions_.size();
      x_square_expected /= positions_.size();
      xy_expected /= positions_.size();

      model_->a_[0] = (xy_expected - x_expected * y_expected) /
                  (x_square_expected - x_expected * x_expected);
      model_->b_ = (x_square_expected * y_expected - x_expected * xy_expected) /
                  (x_square_expected - x_expected * x_expected);
      return;

    }

    // trim down samples to avoid alrge memory usage
    size_t step = 1;
    if (training_keys_.size() > desired_training_key_n_) {
      step = training_keys_.size() / desired_training_key_n_;
    }

    std::vector<size_t> useful_feat_index_;
    for (size_t feat_i = 0; feat_i < model_->max_key_length_; feat_i++) {
      double first_val = training_keys_[0][feat_i];
      for (size_t key_i = 0; key_i < training_keys_.size(); key_i += step) {
        if (training_keys_[key_i][feat_i] != first_val) {
          useful_feat_index_.push_back(feat_i);
          break;
        }
      }
    }

    if (training_keys_.size() != 1 && useful_feat_index_.size() == 0) {
      std::cout<<"all feats are the same"<<std::endl;
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
        std::cout<<"cannot allocate memory for matrix A or b"<<std::endl;
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
          std::cout<<"impossible! cannot fail when there is only 1 bias column in matrix a"
            <<std::endl;
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
    }
    assert(fitting_res == 0);
  }

 private:
  std::vector<double *> training_keys_;
  std::vector<int> positions_;
};

/*** Comparison ***/

struct AlexCompare {
  bool operator()(const AlexKey& x, const AlexKey& y) const {
    //static_assert(
    //    std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
    //    "Comparison types must be numeric.");
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
}