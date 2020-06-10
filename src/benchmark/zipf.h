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

  // FNV hash from https://create.stephan-brumme.com/fnv-hash/
  static const uint32_t PRIME = 0x01000193;  //   16777619
  static const uint32_t SEED = 0x811C9DC5;   // 2166136261
  /// hash a single byte
  inline uint32_t fnv1a(unsigned char oneByte, uint32_t hash = SEED) {
    return (oneByte ^ hash) * PRIME;
  }
  /// hash a 32 bit integer (four bytes)
  inline uint32_t fnv1a(int fourBytes, uint32_t hash = SEED) {
    const unsigned char* ptr = (const unsigned char*)&fourBytes;
    hash = fnv1a(*ptr++, hash);
    hash = fnv1a(*ptr++, hash);
    hash = fnv1a(*ptr++, hash);
    return fnv1a(*ptr, hash);
  }
};