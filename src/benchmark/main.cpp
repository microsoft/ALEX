// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Simple benchmark that runs a mixture of point lookups and inserts on ALEX.
 */

#include "../core/alex.h"

#include <iomanip>
#include <string>

#include "flags.h"
#include "utils.h"

// Modify these if running your own workload
#define KEY_TYPE double
#define PAYLOAD_TYPE double

/*
 * Required flags:
 * --keys_file              path to the file that contains keys
 * --keys_file_type         file type of keys_file (options: binary or text)
 * --init_num_keys          number of keys to bulk load with
 * --total_num_keys         total number of keys in the keys file
 * --batch_size             number of operations (lookup or insert) per batch
 *
 * Optional flags:
 * --insert_frac            fraction of operations that are inserts (instead of
 * lookups)
 * --lookup_distribution    lookup keys distribution (options: uniform or zipf)
 * --time_limit             time limit, in minutes
 * --max_key_length         length of key for string type keys.
 * --print_batch_stats      whether to output stats for each batch
 * --print_key_stats        key related stat print
 */
int main(int argc, char* argv[]) {
  auto flags = parse_flags(argc, argv);
  std::string keys_file_path = get_required(flags, "keys_file");
  std::string keys_file_type = get_required(flags, "keys_file_type");
  auto init_num_keys = stoi(get_required(flags, "init_num_keys"));
  auto total_num_keys = stoi(get_required(flags, "total_num_keys"));
  auto batch_size = stoi(get_required(flags, "batch_size"));
  auto insert_frac = stod(get_with_default(flags, "insert_frac", "0.5"));
  std::string lookup_distribution =
      get_with_default(flags, "lookup_distribution", "zipf");
  auto time_limit = stod(get_with_default(flags, "time_limit", "0.5"));
  auto max_key_length = (unsigned int) stoul(get_with_default(flags, "max_key_length", "1"));
  bool print_batch_stats = get_boolean_flag(flags, "print_batch_stats");
  bool print_key_stats = get_boolean_flag(flags, "print_key_stats");

  // Allocation for key containers.
  auto keys = new alex::AlexKey<KEY_TYPE>[total_num_keys];
  for (int i = 0; i < total_num_keys; i++) { 
    keys[i].key_arr_ = new KEY_TYPE[max_key_length]();
    keys[i].max_key_length_ = max_key_length;
  }

  // Read keys from file
  // PROBLEM : USING ASSERT DOESN'T CALL BELOW FUNCTIONS. NEEDTO FIND OUT WHY.
  // ANSWER : Cmake build type is release, assert is not called (...)
  if (keys_file_type == "binary") {
    load_binary_data(keys, total_num_keys, keys_file_path, max_key_length);
  } else if (keys_file_type == "text") {
    load_text_data(keys, total_num_keys, keys_file_path, max_key_length);
  } else {
    std::cerr << "--keys_file_type must be either 'binary' or 'text'"
              << std::endl;
    return 1;
  }

  // Combine bulk loaded keys with randomly generated payloads
  auto values = new std::pair<alex::AlexKey<KEY_TYPE>, PAYLOAD_TYPE>[init_num_keys];
  std::mt19937_64 gen_payload(std::random_device{}());
  for (int i = 0; i < init_num_keys; i++) {
    values[i].first = keys[i];
    values[i].second = static_cast<PAYLOAD_TYPE>(gen_payload());
    if (print_key_stats) {
      std::cout << "will insert key : ";
      for (unsigned int j = 0; j < max_key_length; j++) {
        std::cout << values[i].first.key_arr_[j];
      } 
      std::cout << ", with payload : " << values[i].second << std::endl;
    }
  }

  // Create ALEX and bulk load
  alex::Alex<KEY_TYPE, PAYLOAD_TYPE> index(max_key_length);
  std::sort(values, values + init_num_keys,
            [](auto const& a, auto const& b) {return a.first < b.first;});
  auto bulkload_start_time = std::chrono::high_resolution_clock::now();
  std::cout << "started bulk_load" << std::endl;
  index.bulk_load(values, init_num_keys);
  std::cout << "finished bulk_load" << std::endl;
  auto bulkload_end_time = std::chrono::high_resolution_clock::now();
  std::cout << "It took " << std::chrono::duration_cast<std::chrono::nanoseconds>(bulkload_end_time -
            bulkload_start_time).count() << "ns" << std::endl;

  // Run workload
  int i = init_num_keys;
  long long cumulative_inserts = 0;
  long long cumulative_lookups = 0;
  int num_inserts_per_batch = static_cast<int>(batch_size * insert_frac);
  int num_lookups_per_batch = batch_size - num_inserts_per_batch;
  double cumulative_insert_time = 0;
  double cumulative_lookup_time = 0;

  auto workload_start_time = std::chrono::high_resolution_clock::now();
  int batch_no = 0;
  PAYLOAD_TYPE sum = 0;
  std::cout << std::scientific;
  std::cout << std::setprecision(3);
  while (true) {
    batch_no++;

    // Do lookups
    double batch_lookup_time = 0.0;
    if (i > 0) {
      alex::AlexKey<KEY_TYPE>* lookup_keys = nullptr;
      if (lookup_distribution == "uniform") {
        lookup_keys = get_search_keys(keys, i, num_lookups_per_batch);
      } else if (lookup_distribution == "zipf") {
        lookup_keys = get_search_keys_zipf(keys, i, num_lookups_per_batch);
      } else {
        std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                  << std::endl;
        return 1;
      }
      auto lookups_start_time = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < num_lookups_per_batch; j++) {
        alex::AlexKey<KEY_TYPE> key = lookup_keys[j];
        PAYLOAD_TYPE* payload = index.get_payload(key);
        if (print_key_stats) {
          std::cout << "lookup key : ";
          for (unsigned int k = 0; k < max_key_length; k++) {
            std::cout << lookup_keys[j].key_arr_[k];
          }
          std::cout << ", payload : " << *payload << std::endl;
          if (payload) {
            sum += *payload;
          }
        }
      }
      auto lookups_end_time = std::chrono::high_resolution_clock::now();
      batch_lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              lookups_end_time - lookups_start_time)
                              .count();
      cumulative_lookup_time += batch_lookup_time;
      cumulative_lookups += num_lookups_per_batch;
      delete[] lookup_keys;
    }

    // Do inserts
    int num_actual_inserts =
        std::min(num_inserts_per_batch, total_num_keys - i);
    int num_keys_after_batch = i + num_actual_inserts;
    auto inserts_start_time = std::chrono::high_resolution_clock::now();
    for (; i < num_keys_after_batch; i++) {
      index.insert(keys[i], static_cast<PAYLOAD_TYPE>(gen_payload()));
      if (print_key_stats) {
        std::cout << "inserted key : ";
        for (unsigned int j = 0; j < max_key_length; j++) {
          std::cout << keys[i].key_arr_[j];
        }
        std::cout << std::endl;
      }
    }
    auto inserts_end_time = std::chrono::high_resolution_clock::now();
    double batch_insert_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(inserts_end_time -
                                                             inserts_start_time)
            .count();
    cumulative_insert_time += batch_insert_time;
    cumulative_inserts += num_actual_inserts;

    if (print_batch_stats) {
      int num_batch_operations = num_lookups_per_batch + num_actual_inserts;
      double batch_time = batch_lookup_time + batch_insert_time;
      long long cumulative_operations = cumulative_lookups + cumulative_inserts;
      double cumulative_time = cumulative_lookup_time + cumulative_insert_time;
      std::cout << "Batch " << batch_no
                << ", cumulative ops: " << cumulative_operations
                << "\n\tbatch throughput:\t"
                << num_lookups_per_batch / batch_lookup_time * 1e9
                << " lookups/sec,\t"
                << num_actual_inserts / batch_insert_time * 1e9
                << " inserts/sec,\t" << num_batch_operations / batch_time * 1e9
                << " ops/sec"
                << "\n\tcumulative throughput:\t"
                << cumulative_lookups / cumulative_lookup_time * 1e9
                << " lookups/sec,\t"
                << cumulative_inserts / cumulative_insert_time * 1e9
                << " inserts/sec,\t"
                << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
                << std::endl;
    }

    // Check for workload end conditions
    if (num_actual_inserts < num_inserts_per_batch) {
      // End if we have inserted all keys in a workload with inserts
      break;
    }
    double workload_elapsed_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - workload_start_time)
            .count();
    if (workload_elapsed_time > time_limit * 1e9 * 60) {
      break;
    }
  }

  long long cumulative_operations = cumulative_lookups + cumulative_inserts;
  double cumulative_time = cumulative_lookup_time + cumulative_insert_time;
  std::cout << "Cumulative stats: " << batch_no << " batches, "
            << cumulative_operations << " ops (" << cumulative_lookups
            << " lookups, " << cumulative_inserts << " inserts)"
            << "\n\tcumulative throughput:\t"
            << cumulative_lookups / cumulative_lookup_time * 1e9
            << " lookups/sec,\t"
            << cumulative_inserts / cumulative_insert_time * 1e9
            << " inserts/sec,\t"
            << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
            << std::endl;

  delete[] values;
  delete[] keys;
}
