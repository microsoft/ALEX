// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gtest/gtest.h"

#define private public
#include "alex.h"

using namespace alex;

namespace test {

TEST(Alex, TestBulkLoad) {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  for (int i = 0; i < 500; i++) {
    values[i].first = rand() % 5000;
    values[i].second = i;
  }

  std::sort(values, values + 500, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 500);

  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());
  }

  EXPECT_EQ(index.get_stats().num_keys, 500);
  index.clear();
  EXPECT_EQ(index.get_stats().num_keys, 0);
}

TEST(Alex, TestConstructors) {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  for (int i = 0; i < 500; i++) {
    values[i].first = rand() % 5000;
    values[i].second = i;
  }

  std::sort(values, values + 500, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 500);

  Alex<int, int> index2(index);  // Copy constructor
  Alex<int, int> index3;
  index3 = index;  // Assignment
  Alex<int, int> index4(std::begin(values), std::end(values));

  EXPECT_NE(index.root_node_, index2.root_node_);
  EXPECT_NE(index.root_node_, index3.root_node_);

  for (int i = 0; i < 500; i++) {
    auto it2 = index2.find(values[i].first);
    EXPECT_TRUE(!it2.is_end());
    EXPECT_EQ(values[i].first, it2.key());

    auto it3 = index3.find(values[i].first);
    EXPECT_TRUE(!it3.is_end());
    EXPECT_EQ(values[i].first, it3.key());

    auto it4 = index4.find(values[i].first);
    EXPECT_TRUE(!it4.is_end());
    EXPECT_EQ(values[i].first, it4.key());
  }
}

TEST(Alex, TestIterators) {
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

  std::sort(values, values + 500, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 500);

  // Iterator from beginning to end
  int num_keys = 0;
  for (auto it = index.begin(); it != index.end(); ++it) {
    num_keys++;
  }
  EXPECT_EQ(500, num_keys);

  // Const iterator from beginning to end
  num_keys = 0;
  for (auto it = index.cbegin(); it != index.cend(); ++it) {
    num_keys++;
  }
  EXPECT_EQ(500, num_keys);

  // Reverse iterator from beginning to end
  num_keys = 0;
  for (auto it = index.rbegin(); it != index.rend(); ++it) {
    num_keys++;
  }
  EXPECT_EQ(500, num_keys);

  // Const reverse iterator from beginning to end
  num_keys = 0;
  for (auto it = index.crbegin(); it != index.crend(); ++it) {
    num_keys++;
  }
  EXPECT_EQ(500, num_keys);

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
  EXPECT_EQ(501, num_keys);

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
  EXPECT_EQ(501, num_keys);
}

TEST(Alex, TestFind) {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  std::sort(values, values + 500, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 500);

  // Find existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());

    int* p = index.get_payload(values[i].first);
    EXPECT_TRUE(p);
    EXPECT_EQ(values[i].second, *p);
  }

  // Find non-existent keys
  for (int i = 1; i < 100; i += 2) {
    auto it = index.find(i);
    EXPECT_TRUE(it.is_end());

    int* p = index.get_payload(i);
    EXPECT_TRUE(!p);
  }
}

// Also tests count and equal_range
TEST(Alex, TestLowerUpperBound) {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  // 10 each of 0, 10, ..., 90
  for (int i = 0; i < 100; i += 10) {
    for (int j = 0; j < 10; j++) {
      values[i + j].first = i;
      values[i + j].second = 0;
    }
  }

  std::sort(values, values + 100, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 100);

  // Search for existent keys
  for (int i = 0; i < 100; i += 10) {
    auto it_lb = index.lower_bound(i);
    auto it_ub = index.upper_bound(i);
    EXPECT_TRUE(!it_lb.is_end());
    EXPECT_EQ(i, it_lb.key());
    if (i == 90) {
      EXPECT_TRUE(it_ub.is_end());
    } else {
      EXPECT_TRUE(!it_ub.is_end());
      EXPECT_EQ(i + 10, it_ub.key());
    }

    // Count
    size_t count = index.count(i);
    EXPECT_EQ(count, 10);

    // Equal range
    auto it_pair = index.equal_range(i);
    EXPECT_TRUE(it_pair.first == it_lb);
    EXPECT_TRUE(it_pair.second == it_ub);
  }

  // Search for non-existent keys
  for (int i = -5; i <= 95; i += 10) {
    auto it_lb = index.lower_bound(i);
    auto it_ub = index.upper_bound(i);
    if (i > 90) {
      EXPECT_TRUE(it_lb.is_end());
      EXPECT_TRUE(it_ub.is_end());
    } else {
      EXPECT_TRUE(!it_lb.is_end());
      EXPECT_TRUE(!it_ub.is_end());
      EXPECT_EQ(i + 5, it_lb.key());
      EXPECT_EQ(i + 5, it_ub.key());
    }

    // Count
    EXPECT_EQ(index.count(i), 0);

    // Equal range
    auto it_pair = index.equal_range(i);
    EXPECT_TRUE(it_pair.first == it_lb);
    EXPECT_TRUE(it_lb == it_ub);
  }
}

TEST(Alex, TestFindLastNoGreaterThan) {
  Alex<int, int> index;

  Alex<int, int>::V values[500];
  // even numbers from 0 to 998 inclusive
  for (int i = 0; i < 500; i++) {
    values[i].first = i * 2;
    values[i].second = i;
  }

  std::sort(values, values + 500, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 500);

  // Existent keys
  for (int i = 0; i < 500; i++) {
    auto it = index.find_last_no_greater_than(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());

    int* p = index.get_payload_last_no_greater_than(values[i].first);
    EXPECT_TRUE(p);
    EXPECT_EQ(values[i].second, *p);
  }

  // Non-existent keys
  for (int i = 0; i < 500; i++) {
    int key = values[i].first + 1;
    auto it = index.find_last_no_greater_than(key);
    EXPECT_TRUE(!it.is_end());
    EXPECT_LE(it.key(), key);
    it++;
    if (!it.is_end()) {
      EXPECT_GT(it.key(), key);
    }

    int* p = index.get_payload_last_no_greater_than(key);
    EXPECT_TRUE(p);
    EXPECT_EQ(values[i].second, *p);
  }

  // Non-existent key smaller than min
  auto it = index.find_last_no_greater_than(-1);
  EXPECT_TRUE(!it.is_end());
  EXPECT_EQ(values[0].first, it.key());
}

TEST(Alex, TestReadModifyWrite) {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = i;
    values[i].second = 0;
  }

  std::sort(values, values + 100, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 100);

  auto it = index.find(50);
  EXPECT_TRUE(!it.is_end());
  EXPECT_EQ(50, it.key());
  EXPECT_EQ(0, it.payload());

  it.payload() = 50;

  it = index.find(50);
  EXPECT_TRUE(!it.is_end());
  EXPECT_EQ(50, it.key());
  EXPECT_EQ(50, it.payload());
}

TEST(Alex, TestSequentialInserts) {
  Alex<int, int> index;

  Alex<int, int>::V values[50];
  for (int i = 0; i < 50; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 50, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 50);

  for (int i = 50; i < 200; i++) {
    auto ret = index.insert(i, i);
    EXPECT_EQ(ret.first.key(), i);
  }

  for (int i = 0; i < 200; i++) {
    auto it = index.find(i);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(i, it.key());
  }
}

TEST(Alex, TestOrderedInserts) {
  Alex<int, int> index;

  Alex<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = i;
  }

  std::sort(values, values + 100, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 100);

  for (int i = 0; i < 100; i++) {
    auto ret = index.insert(2 * i + 1, i);
    EXPECT_EQ(ret.first.key(), 2 * i + 1);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(i);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(i, it.key());
  }
}

TEST(Alex, TestRandomInserts) {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 25, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 25);

  for (int i = 25; i < 200; i++) {
    auto ret = index.insert(values[i].first, values[i].second);
    EXPECT_EQ(ret.first.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());
  }
}

TEST(Alex, TestInsertFromEmpty) {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  for (int i = 0; i < 200; i++) {
    auto ret = index.insert(values[i].first, values[i].second);
    EXPECT_EQ(ret.first.key(), values[i].first);
  }

  // Check that getting the key is correct.
  for (int i = 0; i < 200; i++) {
    auto it = index.find(values[i].first);
    EXPECT_TRUE(!it.is_end());
    EXPECT_EQ(values[i].first, it.key());
  }
}

TEST(Alex, TestRandomErases) {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 200, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
  index.bulk_load(values, 200);

  // Try to erase a nonexistent key
  EXPECT_EQ(index.erase_one(1000), 0);

  // Erase with key
  for (int i = 0; i < 100; i++) {
    int num_erased = index.erase_one(values[i].first);
    EXPECT_EQ(num_erased, 1);
  }

  // Erase with iterator
  for (int i = 100; i < 200; i++) {
    auto it = index.lower_bound(values[i].first);
    EXPECT_TRUE(!it.is_end());
    index.erase(it);
  }

  EXPECT_EQ(index.stats_.num_keys, 0);
}

TEST(Alex, TestRangeScan) {
  Alex<int, int> index;

  Alex<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200, [](auto const &a, auto const &b) {
    return a.first < b.first;
  });
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

TEST(Alex, TestDebug) {}
};