// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * Simple benchmark that runs a mixture of point lookups and inserts on ALEX.
 */

#include "../core/alex.h"

#include <iomanip>
#include <string>
#include <sys/types.h>
#include <unistd.h>

#include "flags.h"
#include "utils.h"

// Modify these if running your own workload
#define KEY_TYPE char
#define PAYLOAD_TYPE double

//parameter for thread
struct FGParam;
typedef FGParam fg_param_t;
void *run_fg(void *param);

struct FGParam {
  uint32_t thread_id;
};

//for multithreading synchronization
volatile bool running = false;
std::atomic<size_t> ready_threads(0);

//common read-only data used for each threads
alex::AlexKey<KEY_TYPE> *keys = nullptr;
alex::Alex<KEY_TYPE, PAYLOAD_TYPE> *table = nullptr;
uint64_t inserted_range = 0;
double insertion_ratio = 0.0;
std::string lookup_distribution;
uint64_t max_key_length = 1;
bool print_key_stats = false;
int total_num_keys = 1;
uint32_t td_num = 1;
uint64_t num_actual_ops_perth;
uint64_t num_actual_lookups_perth;
uint64_t num_actual_inserts_perth;

/*
 * Required flags:
 * --keys_file              path to the file that contains keys
 * --keys_file_type         file type of keys_file (options: binary or text)
 * --init_num_keys          number of keys to bulk load with
 * --total_num_keys         total number of keys in the keys file
 * --batch_size             number of operations (lookup or insert) per batch for all threads
 * --thread_num             number of threads
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
  total_num_keys = stoi(get_required(flags, "total_num_keys"));
  auto batch_size = stoi(get_required(flags, "batch_size"));
  td_num = stoi(get_required(flags, "thread_num"));
  auto insert_frac = stod(get_with_default(flags, "insert_frac", "0.5"));
  lookup_distribution =
      get_with_default(flags, "lookup_distribution", "zipf");
  auto time_limit = stod(get_with_default(flags, "time_limit", "0.5"));
  max_key_length = (unsigned int) stoul(get_with_default(flags, "max_key_length", "1"));
  bool print_batch_stats = get_boolean_flag(flags, "print_batch_stats");
  print_key_stats = get_boolean_flag(flags, "print_key_stats");

  // Allocation for key containers.
  keys = new alex::AlexKey<KEY_TYPE>[total_num_keys];
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

  //extra setup in case of string key
  std::pair<alex::AlexKey<KEY_TYPE>, alex::AtomicVal<PAYLOAD_TYPE>> *values;
  std::mt19937_64 gen_payload(std::random_device{}());
  if (typeid(KEY_TYPE) != typeid(char)) { //numeric
    values = new std::pair<alex::AlexKey<KEY_TYPE>, alex::AtomicVal<PAYLOAD_TYPE>>[init_num_keys];
  }
  else { //string
    values = new std::pair<alex::AlexKey<KEY_TYPE>, alex::AtomicVal<PAYLOAD_TYPE>>[init_num_keys + 2];
    values[init_num_keys].first = alex::AlexKey<KEY_TYPE>(max_key_length);
    values[init_num_keys+1].first = alex::AlexKey<KEY_TYPE>(max_key_length);
    values[init_num_keys].first.key_arr_[0] = STR_VAL_MIN;
    for (unsigned int i = 0; i < max_key_length; i++) {
      values[init_num_keys+1].first.key_arr_[i] = STR_VAL_MAX;
    }
    values[init_num_keys].second = static_cast<alex::AtomicVal<PAYLOAD_TYPE>>(gen_payload());
    values[init_num_keys+1].second = static_cast<alex::AtomicVal<PAYLOAD_TYPE>>(gen_payload());
  }

  // Combine bulk loading keys with randomly generated payloads
  for (int i = 0; i < init_num_keys; i++) {
    values[i].first = keys[i];
    values[i].second = static_cast<alex::AtomicVal<PAYLOAD_TYPE>>(gen_payload());
    if (print_key_stats) {
      std::cout << "will insert key : ";
      for (unsigned int j = 0; j < max_key_length; j++) {
        std::cout << values[i].first.key_arr_[j];
      } 
      std::cout << ", with payload : " << values[i].second.val_ << std::endl;
    }
  }

  if (typeid(KEY_TYPE) == typeid(char)) {init_num_keys += 2;}

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

  //workload setup
  int inserted_range = (typeid(KEY_TYPE) == typeid(char)) ? init_num_keys-2 : init_num_keys;
  int num_inserts_per_batch = static_cast<int>(batch_size * insert_frac);
  int num_lookups_per_batch = batch_size - num_inserts_per_batch;
  table = &index;
  insertion_ratio = insert_frac;

  auto workload_start_time = std::chrono::high_resolution_clock::now();
  int batch_no = 0;
  double cumulative_time = 0.0;
  long long cumulative_operations = 0;
  std::cout << std::scientific;
  std::cout << std::setprecision(3);

  // Run workload
  while (true) {
    batch_no++;
    //Each threads will do lookup/insert randomly for different part,
    //while following insertion ratio.

    //initialize threads
    pthread_t threads[td_num];
    fg_param_t fg_params[td_num];
    running = false;
    for (size_t worker_i = 0; worker_i < td_num; worker_i++) {
      fg_params[worker_i].thread_id = worker_i;
      int ret = pthread_create(&threads[worker_i], nullptr, run_fg,
                              (void *)&fg_params[worker_i]);
      if (ret) {
        std::cout << "Error when making threads with code : " << ret << std::endl;
        abort();
      }
    }
    num_actual_lookups_perth = num_lookups_per_batch / td_num;
    num_actual_inserts_perth = std::min(num_inserts_per_batch / td_num , (total_num_keys - inserted_range) / td_num);
    num_actual_ops_perth = num_actual_lookups_perth + num_actual_inserts_perth;

    while(ready_threads < td_num) {sleep(1);}
    auto batch_start_time = std::chrono::high_resolution_clock::now();
    running = true;

    void *status;
    for (size_t i = 0; i < td_num; i++) {
      int rc = pthread_join(threads[i], &status);
      if (rc) {
        std::cout << "Error : unable to join, " << rc << std::endl;
        abort();
      }
    }

    running = false;
    auto batch_end_time = std::chrono::high_resolution_clock::now();
    inserted_range += num_actual_inserts_perth * td_num;

    if (print_batch_stats) {
      int num_batch_operations = num_lookups_per_batch + num_actual_inserts_perth * td_num;
      double batch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            batch_end_time - batch_start_time)
                            .count();
      cumulative_operations += num_actual_ops_perth * td_num;
      cumulative_time += batch_time;
      std::cout << "Batch " << batch_no
                << ", cumulative ops: " << cumulative_operations
                << "\n\tbatch throughput:\t"
                << num_batch_operations / batch_time * 1e9 << " ops/sec"
                << "\n\tcumulative throughput:\t"
                << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
                << std::endl;
    }

    // Check for workload end conditions
    if (inserted_range >= total_num_keys) {
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

  std::cout << "Cumulative stats: " << batch_no << " batches, "
            << cumulative_operations << " ops"
            << "\n\tcumulative throughput:\t"
            << cumulative_operations / cumulative_time * 1e9 << " ops/sec"
            << std::endl;

  delete[] values;
  delete[] keys;
}


void *run_fg(void *param) {
  //choose which keys to lookup
  FGParam *fgparam = (FGParam *) param;
  uint32_t thread_id = fgparam->thread_id;
  uint64_t insertion_index = inserted_range + (thread_id * num_actual_inserts_perth);

  std::mt19937_64 gen_payload(std::random_device{}());
  alex::AlexKey<KEY_TYPE>* lookup_keys = nullptr;

  //lookup setup
  if (inserted_range > 0) {
    if (lookup_distribution == "uniform") {
      lookup_keys = get_search_keys(keys, inserted_range, num_actual_lookups_perth);
    } else if (lookup_distribution == "zipf") {
      lookup_keys = get_search_keys_zipf(keys, inserted_range, num_actual_lookups_perth);
    } else {
      std::cerr << "--lookup_distribution must be either 'uniform' or 'zipf'"
                << std::endl;
      pthread_exit(nullptr);
    }
  }

  uint64_t insert_cnt = 0, read_cnt = 0;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> ratio_dis(0, 1);
  ready_threads++;

  //wait
  while (!running) ;

  //do batch operations.
  for (uint64_t i = 0; i < num_actual_ops_perth; i++) {
    double d = ratio_dis(gen); //randomly choose which operation to do.
    if ((insert_cnt >= num_actual_inserts_perth) || d <= (insertion_ratio)) { //insert
      table->insert(keys[insertion_index], static_cast<alex::AtomicVal<PAYLOAD_TYPE>>(gen_payload()));
      if (print_key_stats) {
        std::cout << "inserted key : ";
        for (unsigned int j = 0; j < max_key_length; j++) {
          std::cout << keys[i].key_arr_[j];
        }
        std::cout << std::endl;
      }
      insert_cnt++;
      insertion_index++;

    }
    else { //read
      alex::AlexKey<KEY_TYPE> key = lookup_keys[read_cnt];
      PAYLOAD_TYPE* payload = table->get_payload(key);
      if (payload && print_key_stats) {
        std::cout << "lookup key : ";
        for (unsigned int k = 0; k < max_key_length; k++) {
          std::cout << key.key_arr_[k];
        }
        std::cout << ", payload : " << *payload << std::endl;
      }
      else if (print_key_stats) {
        std::cout << "failed finding payload for ";
        for (unsigned int k = 0; k < max_key_length; k++) {
          std::cout << key.key_arr_[k];
        }
        std::cout << std::endl;
      }
      read_cnt++;
    }

  }

  delete[] lookup_keys;

  pthread_exit(nullptr);
}