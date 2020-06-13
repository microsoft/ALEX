// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gtest/gtest.h"

#define private public
#include "alex_multimap.h"

using namespace alex;

namespace test {

TEST(AlexMultimap, TestFind) {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  std::sort(values, values + 500,
            [](auto const &a, auto const &b) { return a.first < b.first; });
  index.bulk_load(values, 500);

  // Find existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());
  }

  // Find non-existent keys
  for (int i = 1; i < 100; i += 2) {
    auto it = index.find(i);
    EXPECT_TRUE(it.is_end());
  }
}

TEST(AlexMultimap, TestRandomInserts) {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 25,
            [](auto const &a, auto const &b) { return a.first < b.first; });
  index.bulk_load(values, 25);

  for (int i = 25; i < 200; i++) {
    auto it = index.insert(values[i].first, values[i].second);
    EXPECT_EQ(it.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());
  }
}

TEST(AlexMultimap, TestRandomErases) {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200,
            [](auto const &a, auto const &b) { return a.first < b.first; });
  index.bulk_load(values, 200);

  // Try to erase a nonexistent key
  EXPECT_EQ(index.erase(1000), 0);

  // Erase with key
  for (int i = 0; i < 100; i++) {
    int num_erased = index.erase(values[i].first);
    EXPECT_EQ(num_erased, 1);
  }

  // Erase with iterator
  for (int i = 100; i < 200; i++) {
    auto it = index.lower_bound(values[i].first);
    EXPECT_TRUE(!it.is_end());
    index.erase(it);
  }

  EXPECT_EQ(index.get_stats().num_keys, 0);
}

TEST(AlexMultimap, TestRangeScan) {
  AlexMultimap<int, int> index;

  AlexMultimap<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200,
            [](auto const &a, auto const &b) { return a.first < b.first; });
  index.bulk_load(values, 200);

  std::vector<int> results;
  int sum = 0;
  auto it = index.begin();
  for (; it != index.end(); it++) {
    results.push_back((*it).second);
    sum += (*it).second;
  }
  EXPECT_EQ(results.size(), 200);
  EXPECT_EQ(sum, 19900);

  std::vector<int> results2;
  int sum2 = 0;
  auto it2 = index.find(10);
  auto it_end = index.find(100);
  for (; it2 != it_end; it2++) {
    results2.push_back((*it2).second);
    sum2 += (*it2).second;
  }
  EXPECT_EQ(results2.size(), 90);
  EXPECT_EQ(sum2, 4905);
}
}