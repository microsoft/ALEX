// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "doctest.h"

#include "alex_multimap.h"

using namespace alex;

TEST_SUITE("AlexMultimap") {

TEST_CASE("TestFind") {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  std::sort(values, values + 500);
  index.bulk_load(values, 500);

  // Find existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }

  // Find non-existent keys
  for (int i = 1; i < 100; i += 2) {
    auto it = index.find(i);
    CHECK(it.is_end());
  }
}

TEST_CASE("TestRandomInserts") {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 25);
  index.bulk_load(values, 25);

  for (int i = 25; i < 200; i++) {
    auto it = index.insert(values[i].first, values[i].second);
    CHECK_EQ(it.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }
}

TEST_CASE("TestRandomErases") {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  index.bulk_load(values, 200);

  // Try to erase a nonexistent key
  CHECK_EQ(index.erase(1000), 0);

  // Erase with key
  for (int i = 0; i < 100; i++) {
    int num_erased = index.erase(values[i].first);
    CHECK_EQ(num_erased, 1);
  }

  // Erase with iterator
  for (int i = 100; i < 200; i++) {
    auto it = index.lower_bound(values[i].first);
    CHECK(!it.is_end());
    index.erase(it);
  }

  CHECK_EQ(index.get_stats().num_keys, 0);
}

TEST_CASE("TestRangeScan") {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  index.bulk_load(values, 200);

  std::vector<int> results;
  int sum = 0;
  for (auto it = index.begin(); it != index.end(); it++) {
    results.push_back((*it).second);
    sum += (*it).second;
  }
  CHECK_EQ(results.size(), 200);
  CHECK_EQ(sum, 19900);

  std::vector<int> results2;
  int sum2 = 0;
  for (auto it = index.find(10), it_end = index.find(100); it != it_end; it++) {
    results2.push_back((*it).second);
    sum2 += (*it).second;
  }
  CHECK_EQ(results2.size(), 90);
  CHECK_EQ(sum2, 4905);
}
}