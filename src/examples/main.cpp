// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This short sample program demonstrates ALEX's API.
 */

#include "../core/alex.h"

#define KEY_TYPE int
#define PAYLOAD_TYPE int

int main(int, char**) {
  // Create some synthetic data: keys are dense integers between 0 and 99, and
  // payloads are random values
  const int num_keys = 100;
  std::pair<KEY_TYPE, PAYLOAD_TYPE> values[num_keys];
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<PAYLOAD_TYPE> dis;
  for (int i = 0; i < num_keys; i++) {
    values[i].first = i;
    values[i].second = dis(gen);
  }

  alex::Alex<KEY_TYPE, PAYLOAD_TYPE> index;

  // Bulk load the keys [0, 100)
  index.bulk_load(values, num_keys);

  // Insert the keys [100, 200). Now there are 200 keys.
  for (int i = num_keys; i < 2 * num_keys; i++) {
    KEY_TYPE new_key = i;
    PAYLOAD_TYPE new_payload = dis(gen);
    index.insert(new_key, new_payload);
  }

  // Erase the keys [0, 10). Now there are 190 keys.
  for (int i = 0; i < 10; i++) {
    index.erase(i);
  }

  // Iterate through all entries in the index and update their payload if the
  // key is even
  int num_entries = 0;
  for (auto it = index.begin(); it != index.end(); it++) {
    if (it.key() % 2 == 0) {  // it.key() is equivalent to (*it).first
      it.payload() = dis(gen);
    }
    num_entries++;
  }
  if (num_entries != 190) {
    std::cout << "Error! There should be 190 entries in the index."
              << std::endl;
  }

  // Iterate through all entries with keys between 50 (inclusive) and 100
  // (exclusive)
  num_entries = 0;
  for (auto it = index.lower_bound(50); it != index.lower_bound(100); it++) {
    num_entries++;
  }
  if (num_entries != 50) {
    std::cout
        << "Error! There should be 50 entries with keys in the range [50, 100)."
        << std::endl;
  }

  // Equivalent way of iterating through all entries with keys between 50
  // (inclusive) and 100 (exclusive)
  num_entries = 0;
  auto it = index.lower_bound(50);
  while (it.key() < 100 && it != index.end()) {
    num_entries++;
    it++;
  }
  if (num_entries != 50) {
    std::cout
        << "Error! There should be 50 entries with keys in the range [50, 100)."
        << std::endl;
  }

  // Insert 9 more keys with value 42. Now there are 199 keys.
  for (int i = 0; i < 9; i++) {
    KEY_TYPE new_key = 42;
    PAYLOAD_TYPE new_payload = dis(gen);
    index.insert(new_key, new_payload);
  }

  // Iterate through all 10 entries with keys of value 42
  int num_duplicates = 0;
  for (auto it = index.lower_bound(42); it != index.upper_bound(42); it++) {
    num_duplicates++;
  }
  if (num_duplicates != 10) {
    std::cout << "Error! There should be 10 entries with key of value 42."
              << std::endl;
  }

  // Check if a non-existent key exists
  it = index.find(1337);
  if (it != index.end()) {
    std::cout << "Error! Key with value 1337 should not exist." << std::endl;
  }

  // Look at some stats
  auto stats = index.get_stats();
  std::cout << "Final num keys: " << stats.num_keys
            << std::endl;  // expected: 199
  std::cout << "Num inserts: " << stats.num_inserts
            << std::endl;  // expected: 109
}