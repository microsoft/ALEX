// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Zipf generator, inspired by
// https://github.com/brianfrankcooper/YCSB/blob/master/core/src/main/java/site/ycsb/generator/ScrambledZipfianGenerator.java
// https://github.com/brianfrankcooper/YCSB/blob/master/core/src/main/java/site/ycsb/generator/ZipfianGenerator.java

class ScrambledZipfianGenerator {
 public:
  static constexpr double ZETAN = 26.46902820178302;
  static constexpr double ZIPFIAN_CONSTANT = 0.99;

  int num_keys_;
  double alpha_;
  double eta_;
  std::mt19937_64 gen_;
  std::uniform_real_distribution<double> dis_;

  explicit ScrambledZipfianGenerator(int num_keys)
      : num_keys_(num_keys), gen_(std::random_device{}()), dis_(0, 1) {
    double zeta2theta = zeta(2);
    alpha_ = 1. / (1. - ZIPFIAN_CONSTANT);
    eta_ = (1 - std::pow(2. / num_keys_, 1 - ZIPFIAN_CONSTANT)) /
           (1 - zeta2theta / ZETAN);
  }

  int nextValue() {
    double u = dis_(gen_);
    double uz = u * ZETAN;

    int ret;
    if (uz < 1.0) {
      ret = 0;
    } else if (uz < 1.0 + std::pow(0.5, ZIPFIAN_CONSTANT)) {
      ret = 1;
    } else {
      ret = (int)(num_keys_ * std::pow(eta_ * u - eta_ + 1, alpha_));
    }

    ret = fnv1a(ret) % num_keys_;
    return ret;
  }

  double zeta(long n) {
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
      sum += 1 / std::pow(i + 1, ZIPFIAN_CONSTANT);
    }
    return sum;
  }

// 64-bit FNV-1a hash, consistent with YCSB's scrambled Zipfian generator.
static constexpr uint64_t FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325ULL;
static constexpr uint64_t FNV_PRIME_64 = 0x100000001b3ULL;

inline uint64_t fnv1a(uint64_t value) {
  uint64_t hash = FNV_OFFSET_BASIS_64;

  for (int i = 0; i < 8; ++i) {
    hash ^= static_cast<uint8_t>(value & 0xff);
    hash *= FNV_PRIME_64;
    value >>= 8;
  }

  return hash;
}
};
