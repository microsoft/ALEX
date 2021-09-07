// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "doctest.h"

#include "alex.h"

using namespace alex;

TEST_SUITE("Alex") {

TEST_CASE("TestBulkLoad") {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  for (int i = 0; i < 500; i++) {
    values[i].first = rand() % 5000;
    values[i].second = i;
  }

  std::sort(values, values + 500);
  index.bulk_load(values, 500);

  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }

  CHECK_EQ(index.get_stats().num_keys, 500);
  index.clear();
  CHECK_EQ(index.get_stats().num_keys, 0);
}

TEST_CASE("TestConstructors") {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  for (int i = 0; i < 500; i++) {
    values[i].first = rand() % 5000;
    values[i].second = i;
  }

  std::sort(values, values + 500);
  index.bulk_load(values, 500);

  Alex<int, int> index2(index);  // Copy constructor
  Alex<int, int> index3;
  index3 = index;  // Assignment
  Alex<int, int> index4(std::begin(values), std::end(values));

  CHECK_NE(index.root_node_, index2.root_node_);
  CHECK_NE(index.root_node_, index3.root_node_);

  for (int i = 0; i < 500; i++) {
    auto it2 = index2.find(values[i].first);
    CHECK(!it2.is_end());
    CHECK_EQ(values[i].first, it2.key());

    auto it3 = index3.find(values[i].first);
    CHECK(!it3.is_end());
    CHECK_EQ(values[i].first, it3.key());

    auto it4 = index4.find(values[i].first);
    CHECK(!it4.is_end());
    CHECK_EQ(values[i].first, it4.key());
  }
}

TEST_CASE("TestIterators") {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  for (int i = 0; i < 250; i++) {
    values[i].first = i;
    values[i].second = i;
  }
  for (int i = 250; i < 500; i++) {
    values[i].first = 5 * i;
    values[i].second = i;
  }

  std::sort(values, values + 500);
  index.bulk_load(values, 500);

  // Iterator from beginning to end
  int num_keys = 0;
  for (auto it = index.begin(); it != index.end(); ++it) {
    num_keys++;
  }
  CHECK_EQ(500, num_keys);

  // Const iterator from beginning to end
  num_keys = 0;
  for (auto it = index.cbegin(); it != index.cend(); ++it) {
    num_keys++;
  }
  CHECK_EQ(500, num_keys);

  // Reverse iterator from beginning to end
  num_keys = 0;
  for (auto it = index.rbegin(); it != index.rend(); ++it) {
    num_keys++;
  }
  CHECK_EQ(500, num_keys);

  // Const reverse iterator from beginning to end
  num_keys = 0;
  for (auto it = index.crbegin(); it != index.crend(); ++it) {
    num_keys++;
  }
  CHECK_EQ(500, num_keys);

  // Convert iterator to reverse iterator
  auto it = index.find(values[250].first);
  auto rit = Alex<int, int>::ReverseIterator(it);
  num_keys = 0;
  for (; it != index.end(); ++it) {
    num_keys++;
  }
  for (; rit != index.rend(); ++rit) {
    num_keys++;
  }
  CHECK_EQ(501, num_keys);

  // Convert const iterator to const reverse iterator
  typename Alex<int, int>::ConstIterator cit = index.find(values[250].first);
  auto crit = Alex<int, int>::ConstReverseIterator(cit);
  num_keys = 0;
  for (; cit != index.cend(); ++cit) {
    num_keys++;
  }
  for (; crit != index.crend(); ++crit) {
    num_keys++;
  }
  CHECK_EQ(501, num_keys);
}

TEST_CASE("TestConst") {
  Alex<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  const Alex<int, int> index(std::begin(values), std::end(values));

  // Find existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }
}

TEST_CASE("TestFind") {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
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

    int *p = index.get_payload(values[i].first);
    CHECK(p);
    CHECK_EQ(values[i].second, *p);
  }

  // Find non-existent keys
  for (int i = 1; i < 100; i += 2) {
    auto it = index.find(i);
    CHECK(it.is_end());

    int *p = index.get_payload(i);
    CHECK(!p);
  }
}

// Also tests count and equal_range
TEST_CASE("TestLowerUpperBound") {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  // 10 each of 0, 10, ..., 90
  for (int i = 0; i < 100; i += 10) {
    for (int j = 0; j < 10; j++) {
      values[i + j].first = i;
      values[i + j].second = 0;
    }
  }

  std::sort(values, values + 100);
  index.bulk_load(values, 100);

  // Search for existent keys
  for (int i = 0; i < 100; i += 10) {
    auto it_lb = index.lower_bound(i);
    auto it_ub = index.upper_bound(i);
    CHECK(!it_lb.is_end());
    CHECK_EQ(i, it_lb.key());
    if (i == 90) {
      CHECK(it_ub.is_end());
    } else {
      CHECK(!it_ub.is_end());
      CHECK_EQ(i + 10, it_ub.key());
    }

    // Count
    size_t count = index.count(i);
    CHECK_EQ(count, 10);

    // Equal range
    auto it_pair = index.equal_range(i);
    CHECK_EQ(it_pair.first, it_lb);
    CHECK_EQ(it_pair.second, it_ub);
  }

  // Search for non-existent keys
  for (int i = -5; i <= 95; i += 10) {
    auto it_lb = index.lower_bound(i);
    auto it_ub = index.upper_bound(i);
    if (i > 90) {
      CHECK(it_lb.is_end());
      CHECK(it_ub.is_end());
    } else {
      CHECK(!it_lb.is_end());
      CHECK(!it_ub.is_end());
      CHECK_EQ(i + 5, it_lb.key());
      CHECK_EQ(i + 5, it_ub.key());
    }

    // Count
    CHECK_EQ(index.count(i), 0);

    // Equal range
    auto it_pair = index.equal_range(i);
    CHECK_EQ(it_pair.first, it_lb);
    CHECK_EQ(it_lb, it_ub);
  }
}

TEST_CASE("TestFindLastNoGreaterThan") {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  std::sort(values, values + 500);
  index.bulk_load(values, 500);

  // Existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find_last_no_greater_than(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());

    int *p = index.get_payload_last_no_greater_than(values[i].first);
    CHECK(p);
    CHECK_EQ(values[i].second, *p);
  }

  // Non-existent keys
  for (int i = 0; i < 500; i++) {
    int key = values[i].first + 1;
    auto it = index.find_last_no_greater_than(key);
    CHECK(!it.is_end());
    CHECK_LE(it.key(), key);
    it++;
    if (!it.is_end()) {
      CHECK_GT(it.key(), key);
    }

    int *p = index.get_payload_last_no_greater_than(key);
    CHECK(p);
    CHECK_EQ(values[i].second, *p);
  }

  // Non-existent key smaller than min
  auto it = index.find_last_no_greater_than(-1);
  CHECK(!it.is_end());
  CHECK_EQ(values[0].first, it.key());
}

TEST_CASE("TestLargeFindLastNoGreaterThan") {
  Alex<uint64_t, uint64_t> index;
  index.insert(std::make_pair(0ULL, 0ULL));

  const uint64_t keys_per_segment = 80523;
  const uint64_t step_size = 43206176;
  const uint64_t num_segments = 16;
  const uint64_t start_keys[] = {698631712, 658125922, 660826308, 663526694,
                                 666227080, 668927466, 671627852, 674328238,
                                 677028624, 679729010, 682429396, 685129782,
                                 687830168, 690530554, 693230940, 695931326};

  uint64_t max_key = 0;
  uint64_t max_key_value = 0;
  for (uint64_t segment = 0; segment < num_segments; ++segment) {
    uint64_t curr_key = start_keys[segment];
    for (uint64_t i = 0; i < keys_per_segment; ++i) {
      if (curr_key > max_key) {
        max_key = curr_key;
        max_key_value = i + 1;
      }

      index.insert(curr_key, i + 1);
      curr_key += step_size;
    }
  }

  // This key is larger than all keys in the index.
  const uint64_t test_key = 3650322694401;
  CHECK_GT(test_key, max_key);

  auto it = index.find_last_no_greater_than(test_key);
  CHECK(!it.is_end());
  CHECK_EQ(max_key, it.key());

  const uint64_t *p = index.get_payload_last_no_greater_than(test_key);
  CHECK(p);
  CHECK_EQ(max_key_value, *p);
}

TEST_CASE("TestReadModifyWrite") {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = i;
    values[i].second = 0;
  }

  std::sort(values, values + 100);
  index.bulk_load(values, 100);

  auto it = index.find(50);
  CHECK(!it.is_end());
  CHECK_EQ(50, it.key());
  CHECK_EQ(0, it.payload());

  it.payload() = 50;

  it = index.find(50);
  CHECK(!it.is_end());
  CHECK_EQ(50, it.key());
  CHECK_EQ(50, it.payload());
}

TEST_CASE("TestSequentialInserts") {
  Alex<int, int> index;

  Alex<int, int>::V values[50];
  for (int i = 0; i < 50; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 50);
  index.bulk_load(values, 50);

  for (int i = 50; i < 200; i++) {
    auto ret = index.insert(i, i);
    CHECK_EQ(ret.first.key(), i);
  }

  for (int i = 0; i < 200; i++) {
    auto it = index.find(i);
    CHECK(!it.is_end());
    CHECK_EQ(i, it.key());
  }
}

TEST_CASE("TestOrderedInserts") {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = i;
  }

  std::sort(values, values + 100);
  index.bulk_load(values, 100);

  for (int i = 0; i < 100; i++) {
    auto ret = index.insert(2 * i + 1, i);
    CHECK_EQ(ret.first.key(), 2 * i + 1);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(i);
    CHECK(!it.is_end());
    CHECK_EQ(i, it.key());
  }
}

TEST_CASE("TestRandomInserts") {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 25);
  index.bulk_load(values, 25);

  for (int i = 25; i < 200; i++) {
    auto ret = index.insert(values[i].first, values[i].second);
    CHECK_EQ(ret.first.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }
}

TEST_CASE("TestInsertFromEmpty") {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  for (int i = 0; i < 200; i++) {
    auto ret = index.insert(values[i].first, values[i].second);
    CHECK_EQ(ret.first.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    CHECK(!it.is_end());
    CHECK_EQ(values[i].first, it.key());
  }
}

TEST_CASE("TestRandomErases") {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  index.bulk_load(values, 200);

  // Try to erase a nonexistent key
  CHECK_EQ(index.erase_one(1000), 0);

  // Erase with key
  for (int i = 0; i < 100; i++) {
    int num_erased = index.erase_one(values[i].first);
    CHECK_EQ(num_erased, 1);
  }

  // Erase with iterator
  for (int i = 100; i < 200; i++) {
    auto it = index.lower_bound(values[i].first);
    CHECK(!it.is_end());
    index.erase(it);
  }

  CHECK_EQ(index.stats_.num_keys, 0);
}

TEST_CASE("TestRangeScan") {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  index.bulk_load(values, 200);

  std::vector<int> results;
  int sum = 0;
  auto it = index.begin();
  for (; it != index.end(); it++) {
    results.push_back((*it).second);
    sum += (*it).second;
  }
  CHECK_EQ(results.size(), 200);
  CHECK_EQ(sum, 19900);

  std::vector<int> results2;
  int sum2 = 0;
  auto it2 = index.find(10);
  auto it_end = index.find(100);
  for (; it2 != it_end; it2++) {
    results2.push_back((*it2).second);
    sum2 += (*it2).second;
  }
  CHECK_EQ(results2.size(), 90);
  CHECK_EQ(sum2, 4905);
}
};
