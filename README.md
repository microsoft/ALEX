
# Usage

```
# Build using CMake, which creates a new build directory
./build.sh

# Run example program
./build/example

# Run unit tests
./build/test_alex
```

# Simple benchmark
We provide a simple benchmark that measures the throughput of running point lookups and inserts on ALEX.
To run on a synthetic dataset with 1000 normally-distributed keys:
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


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
