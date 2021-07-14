# Introduction

ALEX is a ML-enhanced range index, similar in functionality to a B+ Tree.
Our implementation is a near drop-in replacement for std::map or std::multimap.
You can learn more about ALEX in our [SIGMOD 2020 paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3389711).

### Table of Contents
**[Getting Started](#getting-started)**<br>
**[Design Overview](#design-overview)**<br>
**[API Documentation](#api-documentation)**<br>
**[Contributing](#contributing)**<br>

# Getting Started
ALEX can be used as a header-only library.
All relevant header files are found in [src/core](src/core).
You will need to compile your program with at least the C++14 standard (e.g., `-std=c++14`).

In this repository, we include three programs that you can compile and run:
- An [example program](src/examples/main.cpp) of how to use ALEX.
- A [simple benchmark](src/benchmark/main.cpp) that measures the throughput of running point lookups and inserts on ALEX (explained in detail below).
- [Unit tests](test/unittest_main.cpp).

On Windows, simply load this repository into Visual Studio as a CMake project.
On Linux/Mac, use the following commands:
```
# Build using CMake, which creates a new build directory
./build.sh

# Build using CMake in Debug Mode, which creates a new build_debug directory and defines the macro ‘DEBUG'
./build.sh debug

# Run example program
./build/example

# Run unit tests
./build/test_alex
```

To run the benchmark on a synthetic dataset with 1000 normally-distributed keys:
```
./build/benchmark \
--keys_file=resources/sample_keys.bin \
--keys_file_type=binary \
--init_num_keys=500 \
--total_num_keys=1000 \
--batch_size=1000 \
--insert_frac=0.5
```

However, to observe the true performance of ALEX, we must run on a much larger dataset.
You can download a 200M-key (1.6GB) dataset from [Google Drive](https://drive.google.com/file/d/1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl/view?usp=sharing).
To run one example workload on this dataset:
```
./build/benchmark \
--keys_file=[download location] \
--keys_file_type=binary \
--init_num_keys=10000000 \
--total_num_keys=20000000 \
--batch_size=1000000 \
--insert_frac=0.5 \
--lookup_distribution=zipf \
--print_batch_stats
```

You can also run this benchmark on your own dataset.
Your keys will need to be in either binary format or text format (one key per line).
If the data type of your keys is not `double`, you will need to modify `#define KEY_TYPE double` to
`#define KEY_TYPE [your data type]` in [src/benchmark/main.cpp](src/benchmark/main.cpp).

### Datasets
The four datasets we used in our [SIGMOD 2020 paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3389711) are publicly available (all in binary format):
- [Longitudes (200M 8-byte floats)](https://drive.google.com/file/d/1zc90sD6Pze8UM_XYDmNjzPLqmKly8jKl/view?usp=sharing)
- [Longlat (200M 8-byte floats)](https://drive.google.com/file/d/1mH-y_PcLQ6p8kgAz9SB7ME4KeYAfRfmR/view?usp=sharing)
- [Lognormal (190M 8-byte signed ints)](https://drive.google.com/file/d/1y-UBf8CuuFgAZkUg_2b_G8zh4iF_N-mq/view?usp=sharing)
- [YCSB (200M 8-byte unsigned ints)](https://drive.google.com/file/d/1Q89-v4FJLEwIKL3YY3oCeOEs0VUuv5bD/view?usp=sharing)

# Design Overview
Like the B+ Tree, ALEX is a data structure that indexes sorted data and supports workloads that contain a mix of point lookups, short range queries, inserts, updates, and deletes.
Internally, ALEX uses a collection of linear regressions, organized hierarchically into a tree, to model the distribution of keys.
ALEX uses this model to efficiently search for data records by their key.
ALEX also automatically adapts its internal models and tree structure to efficiently support writes.

ALEX is inspired by the [original learned index from Kraska et al.](https://dl.acm.org/doi/pdf/10.1145/3183713.3196909).
However, that work only supports reads (i.e., point lookups and range queries), while ALEX also efficiently supports write (i.e., inserts, updates, and deletes).

In [our paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3389711), we show that ALEX outperforms alternatives in both speed and size:
- On read-only workloads, ALEX beats the [original learned index from Kraska et al.](https://dl.acm.org/doi/pdf/10.1145/3183713.3196909) by
  up to 2.2X on performance with up to 15X smaller index size.
- Across the spectrum of read-write workloads, ALEX beats
  B+ Trees (implemented by [STX B+ Tree](https://panthema.net/2007/stx-btree/)) by up to 4.1X while never performing worse, with
  up to 2000X smaller index size.

You can find many more details about ALEX [here](https://dl.acm.org/doi/pdf/10.1145/3318464.3389711).

### Limitations and Future Research Directions
ALEX currently operates in memory, single threaded, and on numerical keys.
We are considering ways to add support for persistence, concurrency, and string keys to ALEX. 

In terms of performance, ALEX has a couple of known limitations:
- The premise of ALEX is to model the key distribution using a collection of linear regressions.
Therefore, ALEX performs poorly when the key distribution is difficult to model with linear regressions, i.e., when the key distribution is highly nonlinear at small scales.
A possible future research direction is to use a broader class of modeling techniques (e.g., also consider polynomial regression models).
- ALEX can have poor performance in the presence of extreme outlier keys, which can cause the key domain and ALEX's tree depth to become unnecessarily large
(see Section 5.1 of [our paper](https://dl.acm.org/doi/pdf/10.1145/3318464.3389711)).
A possible future research direction is to add special logic for handling extreme outliers, or to have a modeling strategy that is robust to sparse key spaces.

# API Documentation
We provide three user-facing implementations of ALEX:
1. [AlexMap](https://github.com/microsoft/ALEX/blob/master/src/core/alex_map.h) is a near drop-in replacement for [std::map](http://www.cplusplus.com/reference/map/map/).
2. [AlexMultiMap](https://github.com/microsoft/ALEX/blob/master/src/core/alex_multimap.h) is a near drop-in replacement for [std::multimap](http://www.cplusplus.com/reference/map/multimap/).
3. [Alex](https://github.com/microsoft/ALEX/blob/master/src/core/alex.h) is the internal implementation that supports both AlexMap and AlexMultimap. It exposes slightly more functionality.

ALEX has a few important differences compared to its standard library equivalents:
- Keys and payloads (i.e., the mapped type) are stored separately, so dereferencing an iterator returns a copy of the key/payload pair, not a reference.
Our iterators have methods to directly return references to the key or payload individually.
- The iterators are of type ForwardIterator, instead of BidirectionalIterator.
Therefore, iterators do not support decrementing.
- Currently, we only support numerical key types.
We do not support arbitrary user-defined key types.
As a result, you should not change the default comparison function in the class template.

Detailed API documentation can be found [in our wiki](https://github.com/microsoft/ALEX/wiki/API-Documentation).

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
