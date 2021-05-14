// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "doctest.h"

#include "alex_nodes.h"

using namespace alex;

TEST_SUITE("DataNode") {

/************************* Tests for Data Node *****************************/

TEST_CASE("TestBinarySearch") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = int(0.2 * i) * 2;
    values[i].second = 0;
  }

  std::vector<int> keys_to_search;
  for (int i = 0; i < 40; i++) {
    keys_to_search.push_back(i);
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  for (int key : keys_to_search) {
    int lower_bound_pos =
        node.binary_search_lower_bound(0, node.data_capacity_, key);
    if (lower_bound_pos > 0) {
      CHECK_LT(node.get_key(lower_bound_pos - 1), key);
    }
    if (lower_bound_pos < node.data_capacity_) {
      CHECK_GE(node.get_key(lower_bound_pos), key);
    }

    int upper_bound_pos =
        node.binary_search_upper_bound(0, node.data_capacity_, key);
    if (upper_bound_pos > 0) {
      CHECK_LE(node.get_key(upper_bound_pos - 1), key);
    }
    if (upper_bound_pos < node.data_capacity_) {
      CHECK_GT(node.get_key(upper_bound_pos), key);
    }
  }
}

TEST_CASE("TestExponentialSearch") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = int(0.2 * i) * 2;
    values[i].second = 0;
  }

  std::vector<int> keys_to_search;
  for (int i = 0; i < 40; i++) {
    keys_to_search.push_back(i);
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  for (int key : keys_to_search) {
    for (int m = 0; m < node.data_capacity_; m++) {
      int lower_bound_pos = node.exponential_search_lower_bound(m, key);
      if (lower_bound_pos > 0) {
        CHECK_LT(node.get_key(lower_bound_pos - 1), key);
      }
      if (lower_bound_pos < node.data_capacity_) {
        CHECK_GE(node.get_key(lower_bound_pos), key);
      }

      int upper_bound_pos = node.exponential_search_upper_bound(m, key);
      if (upper_bound_pos > 0) {
        CHECK_LE(node.get_key(upper_bound_pos - 1), key);
      }
      if (upper_bound_pos < node.data_capacity_) {
        CHECK_GT(node.get_key(upper_bound_pos), key);
      }
    }
  }
}

TEST_CASE("TestNumKeysInRange") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  int num_keys = node.num_keys_in_range(0, node.data_capacity_);
  CHECK_EQ(num_keys, 100);

  int num_keys_first_half = node.num_keys_in_range(0, node.data_capacity_ / 2);
  int num_keys_second_half =
      node.num_keys_in_range(node.data_capacity_ / 2, node.data_capacity_);
  CHECK_EQ(num_keys, num_keys_first_half + num_keys_second_half);
}

TEST_CASE("TestNextFilledPosition") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  for (int i = 0; i < node.data_capacity_; i++) {
    int next_filled_pos = node.get_next_filled_position(i, true);
    CHECK_LT(i, next_filled_pos);
    if (next_filled_pos < node.data_capacity_) {
      CHECK(node.check_exists(next_filled_pos));
    }
    for (int j = i + 1; j < next_filled_pos; j++) {
      CHECK(!node.check_exists(j));
    }
  }
}

TEST_CASE("TestClosestGap") {
  auto node = new AlexDataNode<int, int>();

#if ALEX_DATA_NODE_SEP_ARRAYS
  node->key_slots_ = new int[8]{1, 1, 2, 3, 4, 5, 6, 6};
  node->payload_slots_ = new int[8]();
#else
  node->data_slots_ = new std::pair<int, int>[8]{
      {1, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {6, 0}};
#endif
  node->data_capacity_ = 8;
  node->bitmap_ = new uint64_t[1]();
  for (int i : {1, 2, 3, 4, 5, 7}) {
    node->bitmap_[0] |= (1ULL << i);
  }

  CHECK_EQ(0, node->closest_gap(1));
  CHECK_EQ(0, node->closest_gap(2));
  CHECK_EQ(0,
            node->closest_gap(3));  // border case, by default choose left gap
  CHECK_EQ(6, node->closest_gap(4));
  CHECK_EQ(6, node->closest_gap(5));
  CHECK_EQ(6, node->closest_gap(7));
  delete node;

  // Test with 5 bitmap blocks
  node = new AlexDataNode<int, int>();
#if ALEX_DATA_NODE_SEP_ARRAYS
  node->key_slots_ = new int[320];
  node->payload_slots_ = new int[320]();
  for (int i = 0; i < 50; i++) {
    node->key_slots_[i] = 50;
  }
  for (int i = 50; i < 300; i++) {
    node->key_slots_[i] = i;
  }
  for (int i = 300; i < 320; i++) {
    node->key_slots_[i] = AlexDataNode<int, int>::kEndSentinel_;
  }
#else
  node->data_slots_ = new std::pair<int, int>[192];
  for (int i = 0; i < 50; i++) {
    node->data_slots_[i] = std::pair<int, int>(50, 0);
  }
  for (int i = 50; i < 300; i++) {
    node->data_slots_[i] = std::pair<int, int>(i, 0);
  }
  for (int i = 300; i < 320; i++) {
    node->data_slots_[i] =
        std::pair<int, int>(AlexDataNode<int, int>::kEndSentinel, 0);
  }
#endif
  node->data_capacity_ = 320;
  node->bitmap_ = new uint64_t[5]();
  for (int i = 50; i < 300; i++) {
    size_t bitmap_pos = i >> 6;
    size_t bit_pos = i - (bitmap_pos << 6);
    node->bitmap_[bitmap_pos] |= (1ULL << bit_pos);
  }

  CHECK_EQ(49, node->closest_gap(75));    // pos in second block
  CHECK_EQ(49, node->closest_gap(130));   // pos in third block
  CHECK_EQ(300, node->closest_gap(180));  // pos in third block
  CHECK_EQ(300, node->closest_gap(200));  // pos in fourth block
  delete node;
}

TEST_CASE("TestInsertUsingShifts") {
  AlexDataNode<int, int> node;

#if ALEX_DATA_NODE_SEP_ARRAYS
  node.key_slots_ = new int[8]{1, 1, 2, 3, 5, 6, 7, 7};
  node.payload_slots_ = new int[8]();
#else
  node.data_slots_ = new std::pair<int, int>[8]{{1, 0}, {1, 0}, {2, 0}, {3, 0},
                                                {5, 0}, {6, 0}, {7, 0}, {7, 0}};
#endif
  node.data_capacity_ = 8;
  node.bitmap_ = new uint64_t[1]();
  for (int i : {1, 2, 3, 4, 5, 7}) {
    node.bitmap_[0] |= (1ULL << i);
  }

  node.insert_using_shifts(4, rand(), 4);
  int expected[] = {1, 1, 2, 3, 4, 5, 6, 7};
  for (int i = 0; i < 8; i++) {
    CHECK_EQ(expected[i], node.get_key(i));
  }
}

TEST_CASE("TestCheckExists") {
  AlexDataNode<int, int> node;

#if ALEX_DATA_NODE_SEP_ARRAYS
  node.key_slots_ = new int[8]{1, 1, 2, 3, 4, 5, 6, 6};
  node.payload_slots_ = new int[8]();
#else
  node.data_slots = new std::pair<int, int>[8]{{1, 0}, {1, 0}, {2, 0}, {3, 0},
                                               {4, 0}, {5, 0}, {6, 0}, {6, 0}};
#endif
  node.data_capacity_ = 8;
  node.bitmap_ = new uint64_t[1]();
  bool exists[] = {false, true, true, true, true, true, false, true};
  for (int i = 0; i < 8; i++) {
    if (exists[i]) {
      node.bitmap_[0] |= (1ULL << i);
    }
  }

  for (int i = 0; i < 8; i++) {
    CHECK_EQ(exists[i], node.check_exists(i));
  }
}

TEST_CASE("TestExpansion") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = rand() % 500;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  node.resize(0.5, true);

  for (int i = 0; i < 100; i++) {
    int pos = node.find_key(values[i].first);
    CHECK_EQ(values[i].first, node.get_key(pos));
    CHECK(node.check_exists(pos));
  }

  node.resize(0.9, true);

  for (int i = 0; i < 100; i++) {
    int pos = node.find_key(values[i].first);
    CHECK_EQ(values[i].first, node.get_key(pos));
    CHECK(node.check_exists(pos));
  }
}

TEST_CASE("TestFindInsertPosition") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  for (int key = 0; key < node.data_capacity_; key++) {
    std::pair<int, int> insert_pos = node.find_insert_position(key);
    int model_based_insert_pos = insert_pos.first;
    if (model_based_insert_pos > 0) {
      CHECK_GE(key, node.get_key(model_based_insert_pos - 1));
    }
    if (model_based_insert_pos < node.data_capacity_) {
      CHECK_LE(key, node.get_key(model_based_insert_pos));
    }
  }
}

TEST_CASE("TestIterator") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  std::vector<int> results;
  AlexDataNode<int, int>::const_iterator_type it(&node, 0);
  for (; !it.is_end(); it++) {
    results.push_back(it.key());
  }
  CHECK_EQ(100, results.size());
}

TEST_CASE("TestIteratorWithDuplicates") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = i / 2;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  CHECK_EQ(node.find_upper(10), node.find_lower(11));

  std::vector<int> results;
  AlexDataNode<int, int>::const_iterator_type it(&node, node.find_lower(2));
  AlexDataNode<int, int>::const_iterator_type end_it(&node, node.find_lower(4));
  for (; it != end_it; it++) {
    results.push_back(it.key());
  }
  CHECK_EQ(4, results.size());
}

TEST_CASE("TestBulkLoadFromExisting") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[100];
  for (int i = 0; i < 100; i++) {
    values[i].first = 2 * i;
    values[i].second = rand();
  }

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  AlexDataNode<int, int> new_node;
  new_node.bulk_load_from_existing(&node, 0, node.data_capacity_);

  int key = 50;
  for (int pos = 0; pos < new_node.data_capacity_; pos++) {
    int actual_pos = new_node.exponential_search_upper_bound(pos, key) - 1;
    CHECK_EQ(key, new_node.get_key(actual_pos));
    CHECK(node.check_exists(actual_pos));
  }
}

TEST_CASE("TestInserts") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }
  std::shuffle(values, values + 200, std::default_random_engine{});

  std::sort(values, values + 100);
  node.bulk_load(values, 100);

  for (int i = 100; i < 200; i++) {
    node.insert(values[i].first, values[i].second);
  }

  for (int i = 0; i < 200; i++) {
    int pos = node.find_key(values[i].first);
    CHECK_EQ(values[i].first, node.get_key(pos));
    CHECK(node.check_exists(pos));
  }
}

TEST_CASE("TestInsertsWithDuplicates") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  node.bulk_load(values, 200);

  std::shuffle(values, values + 200, std::default_random_engine{});
  for (int i = 0; i < 200; i++) {
    node.insert(values[i].first, values[i].second);
  }

  for (int i = 0; i < 200; i++) {
    int pos = node.find_key(values[i].first);
    CHECK_EQ(values[i].first, node.get_key(pos));
    CHECK(node.check_exists(pos));
  }
}

TEST_CASE("TestEraseOne") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = rand() % 500;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  node.bulk_load(values, 200);

  for (int i = 0; i < 150; i++) {
    int num_erased = node.erase_one(values[i].first);
    CHECK_EQ(num_erased, 1);
  }

  for (int i = 150; i < 200; i++) {
    int pos = node.find_key(values[i].first);
    CHECK_EQ(values[i].first, node.get_key(pos));
    CHECK(node.check_exists(pos));
    node.erase_one(values[i].first);
  }

  CHECK_EQ(node.num_keys_, 0);
}

TEST_CASE("TestErase") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i / 2;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  node.bulk_load(values, 200);

  for (int i = 0; i < 75; i++) {
    int num_erased = node.erase(i);
    CHECK_EQ(num_erased, 2);
  }

  for (int i = 75; i < 100; i++) {
    int pos = node.find_key(i);
    CHECK_EQ(i, node.get_key(pos));
    CHECK(node.check_exists(pos));
    node.erase(i);
  }

  CHECK_EQ(node.num_keys_, 0);
}

TEST_CASE("TestEraseRange") {
  AlexDataNode<int, int> node;

  AlexDataNode<int, int>::V values[200];
  for (int i = 0; i < 200; i++) {
    values[i].first = i;
    values[i].second = i;
  }

  std::sort(values, values + 200);
  node.bulk_load(values, 200);

  int num_erased = node.erase_range(50, 100);
  CHECK_EQ(num_erased, 50);

  num_erased = node.erase_range(-50, 50);
  CHECK_EQ(num_erased, 50);

  num_erased = node.erase_range(150, 300);
  CHECK_EQ(num_erased, 50);

  for (int i = 100; i < 150; i++) {
    int pos = node.find_key(i);
    CHECK_EQ(i, node.get_key(pos));
    CHECK(node.check_exists(pos));
    node.erase_range(i, i + 1);
  }

  CHECK_EQ(node.num_keys_, 0);
}

TEST_CASE("TestBuildIndexWithSample") {
  const int num_keys = 20000;
  AlexDataNode<int, int>::V values[num_keys];
  for (int i = 0; i < num_keys; i++) {
    values[i].first = rand() % 50000 + 10000;
  }
  std::sort(values, values + num_keys);

  LinearModel<int> model;
  LinearModel<int> model_using_sample;

  AlexDataNode<int, int>::build_model(values, num_keys, &model, false);
  AlexDataNode<int, int>::build_model(values, num_keys, &model_using_sample,
                                      true);

  double rel_diff_in_a =
      std::abs((model.a_ - model_using_sample.a_) / model.a_);
  double rel_diff_in_b =
      std::abs((model.b_ - model_using_sample.b_) / model.b_);
  CHECK_LT(rel_diff_in_a, 0.05);
  CHECK_LT(rel_diff_in_b, 0.05);
}

TEST_CASE("TestComputeCostWithSample") {
  const int num_keys = 20000;
  AlexDataNode<int, int>::V values[num_keys];
  for (int i = 0; i < num_keys; i++) {
    values[i].first = rand() % 50000;
  }
  std::sort(values, values + num_keys);

  LinearModel<int> model;
  AlexDataNode<int, int>::build_model(values, num_keys, &model);
  double density = 0.7;
  double expected_insert_frac = 0.5;
  ExpectedIterationsAndShiftsAccumulator ent;
  ExpectedIterationsAndShiftsAccumulator ent_using_sample;

  DataNodeStats stats;
  DataNodeStats stats_using_sample;
  AlexDataNode<int, int>::compute_expected_cost(
      values, num_keys, density, expected_insert_frac, &model, false, &stats);
  AlexDataNode<int, int>::compute_expected_cost(values, num_keys, density,
                                                expected_insert_frac, &model,
                                                true, &stats_using_sample);

  double exp_iters = stats.num_search_iterations;
  double exp_iters_sample = stats_using_sample.num_search_iterations;
  double exp_shifts = stats.num_shifts;
  double exp_shifts_sample = stats_using_sample.num_shifts;
  double rel_diff_in_search_entropy =
      std::abs((exp_iters - exp_iters_sample) / exp_iters);
  double rel_diff_in_shifts_entropy =
      std::abs((exp_shifts - exp_shifts_sample) / exp_shifts);
  CHECK_LT(rel_diff_in_search_entropy, 0.5);
  CHECK_LT(rel_diff_in_shifts_entropy, 0.5);
}
};