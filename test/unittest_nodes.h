// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "gtest/gtest.h"

#define private public
#include "../src/core/alex_nodes.h"

using namespace alex;

namespace test
{

    /************************* Tests for Data Node *****************************/

    TEST(DataNode, TestBinarySearch)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = int(0.2*i) * 2;
            payload[i] = 0;
        }

        std::vector<int> keys_to_search;
        for (int i = 0; i < 40; i++) {
            keys_to_search.push_back(i);
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        for (int key : keys_to_search) {
            int lower_bound_pos = node.binary_search_lower_bound(0, node.data_capacity_, key);
            if (lower_bound_pos > 0) {
                EXPECT_LT(node.get_key(lower_bound_pos - 1), key);
            }
            if (lower_bound_pos < node.data_capacity_) {
                EXPECT_GE(node.get_key(lower_bound_pos), key);
            }

            int upper_bound_pos = node.binary_search_upper_bound(0, node.data_capacity_, key);
            if (upper_bound_pos > 0) {
                EXPECT_LE(node.get_key(upper_bound_pos - 1), key);
            }
            if (upper_bound_pos < node.data_capacity_) {
                EXPECT_GT(node.get_key(upper_bound_pos), key);
            }
        }
    }

    TEST(DataNode, TestExponentialSearch)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = int(0.2*i) * 2;
            payload[i] = 0;
        }

        std::vector<int> keys_to_search;
        for (int i = 0; i < 40; i++) {
            keys_to_search.push_back(i);
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        for (int key : keys_to_search) {
            for (int m = 0; m < node.data_capacity_; m++) {
                int lower_bound_pos = node.exponential_search_lower_bound(m, key);
                if (lower_bound_pos > 0) {
                    EXPECT_LT(node.get_key(lower_bound_pos - 1), key);
                }
                if (lower_bound_pos < node.data_capacity_) {
                    EXPECT_GE(node.get_key(lower_bound_pos), key);
                }

                int upper_bound_pos = node.exponential_search_upper_bound(m, key);
                if (upper_bound_pos > 0) {
                    EXPECT_LE(node.get_key(upper_bound_pos - 1), key);
                }
                if (upper_bound_pos < node.data_capacity_) {
                    EXPECT_GT(node.get_key(upper_bound_pos), key);
                }
            }
        }
    }

    TEST(DataNode, TestNextFilledPosition)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = 2*i;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        for (int i = 0; i < node.data_capacity_; i++) {
            int next_filled_pos = node.get_next_filled_position(i, true);
            EXPECT_LT(i, next_filled_pos);
            if (next_filled_pos < node.data_capacity_) {
                EXPECT_TRUE(node.check_exists(next_filled_pos));
            }
            for (int j = i + 1; j < next_filled_pos; j++) {
                EXPECT_TRUE(!node.check_exists(j));
            }
        }
    }

    TEST(DataNode, TestClosestGap)
    {
        AlexDataNode<int, int> node;

#if ALEX_DATA_NODE_SEP_ARRAYS
        node.key_slots_ = new int[8]{ 1, 1, 2, 3, 4, 5, 6, 6 };
        node.payload_slots_ = new int[8]();
#else
        node.data_slots_ = new std::pair<int, int>[8]{ {1,0}, {1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {6,0} };
#endif
        node.data_capacity_ = 8;
        node.bitmap_ = new uint64_t[1]();
        for (int i : { 1, 2, 3, 4, 5, 7 }) {
            node.bitmap_[0] |= (1L << i);
        }

        EXPECT_EQ(0, node.closest_gap(1));
        EXPECT_EQ(0, node.closest_gap(2));
        EXPECT_EQ(0, node.closest_gap(3));  // border case, by default choose left gap
        EXPECT_EQ(6, node.closest_gap(4));
        EXPECT_EQ(6, node.closest_gap(5));
        EXPECT_EQ(6, node.closest_gap(7));

        // Test with 5 bitmap blocks
#if ALEX_DATA_NODE_SEP_ARRAYS
        node.key_slots_ = new int[320];
        node.payload_slots_ = new int[320]();
        for (int i = 0; i < 50; i++) {
            node.key_slots_[i] = 50;
        }
        for (int i = 50; i < 300; i++) {
            node.key_slots_[i] = i;
        }
        for (int i = 300; i < 320; i++) {
            node.key_slots_[i] = AlexDataNode<int, int>::kEndSentinel_;
        }
#else
        node.data_slots_ = new std::pair<int, int>[192];
        for (int i = 0; i < 50; i++) {
            node.data_slots_[i] = std::pair<int, int>(50, 0);
        }
        for (int i = 50; i < 300; i++) {
            node.data_slots_[i] = std::pair<int, int>(i, 0);
        }
        for (int i = 300; i < 320; i++) {
            node.data_slots_[i] = std::pair<int, int>(AlexDataNode<int, int>::kEndSentinel, 0);
        }
#endif
        node.data_capacity_ = 320;
        node.bitmap_ = new uint64_t[5]();
        for (int i = 50; i < 300; i++) {
            size_t bitmap_pos = i >> 6;
            size_t bit_pos = i - (bitmap_pos << 6);
            node.bitmap_[bitmap_pos] |= (1L << bit_pos);
        }

        EXPECT_EQ(49, node.closest_gap(75));  // pos in second block
        EXPECT_EQ(49, node.closest_gap(130));  // pos in third block
        EXPECT_EQ(300, node.closest_gap(180));  // pos in third block
        EXPECT_EQ(300, node.closest_gap(200));  // pos in fourth block
    }

    TEST(DataNode, TestInsertUsingShifts)
    {
        AlexDataNode<int, int> node;

#if ALEX_DATA_NODE_SEP_ARRAYS
        node.key_slots_ = new int[8]{ 1, 1, 2, 3, 5, 6, 7, 7 };
        node.payload_slots_ = new int[8]();
#else
        node.data_slots_ = new std::pair<int, int>[8]{ {1,0}, {1,0}, {2,0}, {3,0}, {5,0}, {6,0}, {7,0}, {7,0} };
#endif
        node.data_capacity_ = 8;
        node.bitmap_ = new uint64_t[1]();
        for (int i : { 1, 2, 3, 4, 5, 7 }) {
            node.bitmap_[0] |= (1L << i);
        }

        node.insert_using_shifts(4, rand(), 4, 0, 8);
        int expected[] = { 1, 1, 2, 3, 4, 5, 6, 7 };
        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(expected[i], node.get_key(i));
        }
    }

    TEST(DataNode, TestCheckExists)
    {
        AlexDataNode<int, int> node;

#if ALEX_DATA_NODE_SEP_ARRAYS
        node.key_slots_ = new int[8]{ 1, 1, 2, 3, 4, 5, 6, 6 };
        node.payload_slots_ = new int[8]();
#else
        node.data_slots = new std::pair<int, int>[8]{ {1,0}, {1,0}, {2,0}, {3,0}, {4,0}, {5,0}, {6,0}, {6,0} };
#endif
        node.data_capacity_ = 8;
        node.bitmap_ = new uint64_t[1]();
        bool exists[] = { false, true, true, true, true, true, false, true };
        for (int i = 0; i < 8; i++) {
            if (exists[i]) {
                node.bitmap_[0] |= (1L << i);
            }
        }

        for (int i = 0; i < 8; i++) {
            EXPECT_EQ(exists[i], node.check_exists(i));
        }
    }

    TEST(DataNode, TestExpansion)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = rand() % 500;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        node.resize(0.5, true);

        for (int i = 0; i < 100; i++) {
            int pos = node.find_key(keys[i]);
            EXPECT_EQ(keys[i], node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
        }

        node.resize(0.9, true);

        for (int i = 0; i < 100; i++) {
            int pos = node.find_key(keys[i]);
            EXPECT_EQ(keys[i], node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
        }
    }

    TEST(DataNode, TestFindInsertPosition)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = 2*i;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        for (int key = 0; key < node.data_capacity_; key++) {
            int insert_pos = node.find_insert_position(key);
            if (insert_pos > 0) {
                EXPECT_TRUE(key >= node.get_key(insert_pos - 1));
            }
            if (insert_pos < node.data_capacity_) {
                EXPECT_TRUE(key <= node.get_key(insert_pos));
            }
        }
    }

    TEST(DataNode, TestIterator)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = 2*i;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        std::vector<int> results;
        AlexDataNode<int, int>::Iterator it(&node, 0);
        for (; !it.is_end(); it++) {
            results.push_back(it.key());
        }
        EXPECT_EQ(100, results.size());
    }

    TEST(DataNode, TestIteratorWithDuplicates)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = i / 2;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        EXPECT_TRUE(node.find_upper(10) == node.find_lower(11));

        std::vector<int> results;
        AlexDataNode<int, int>::Iterator it(&node, node.find_lower(2));
        AlexDataNode<int, int>::Iterator end_it(&node, node.find_lower(4));
        for (; it != end_it; it++) {
            results.push_back(it.key());
        }
        EXPECT_EQ(4, results.size());
    }

    TEST(DataNode, TestBulkLoadFromExisting)
    {
        AlexDataNode<int, int> node;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = 2*i;
            payload[i] = rand();
        }

        std::sort(keys, keys + 100);
        node.bulk_load(keys, payload, 100);

        AlexDataNode<int, int> new_node;
        new_node.bulk_load_from_existing(&node, 0, node.data_capacity_);

        int key = 50;
        for (int pos = 0; pos < new_node.data_capacity_; pos++) {
            int actual_pos = new_node.exponential_search_upper_bound(pos, key) - 1;
            EXPECT_EQ(key, new_node.get_key(actual_pos));
            EXPECT_TRUE(node.check_exists(actual_pos));
        }
    }

    TEST(DataNode, TestInserts)
    {
        AlexDataNode<int, int> node;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = i;
            payload[i] = i;
        }
        std::random_shuffle(keys, keys + 200);

        std::sort(keys, keys + 25);
        node.bulk_load(keys, payload, 25);

        for (int i = 25; i < 200; i++) {
            node.insert(keys[i], payload[i]);
        }

        for (int i = 0; i < 200; i++) {
            int pos = node.find_key(keys[i]);
            EXPECT_EQ(keys[i], node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
        }
    }

    TEST(DataNode, TestInsertsWithDuplicates)
    {
        AlexDataNode<int, int> node;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = i;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        node.bulk_load(keys, payload, 200);

        std::random_shuffle(keys, keys + 200);
        for (int i = 0; i < 200; i++) {
            node.insert(keys[i], payload[i]);
        }

        for (int i = 0; i < 200; i++) {
            int pos = node.find_key(keys[i]);
            EXPECT_EQ(keys[i], node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
        }
    }

    TEST(DataNode, TestEraseOne)
    {
        AlexDataNode<int, int> node;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = rand() % 500;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        node.bulk_load(keys, payload, 200);

        for (int i = 0; i < 150; i++) {
            int num_erased = node.erase_one(keys[i]);
            EXPECT_EQ(num_erased, 1);
        }

        for (int i = 150; i < 200; i++) {
            int pos = node.find_key(keys[i]);
            EXPECT_EQ(keys[i], node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
            node.erase_one(keys[i]);
        }

        EXPECT_TRUE(node.num_keys_ == 0);
    }

    TEST(DataNode, TestErase)
    {
        AlexDataNode<int, int> node;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = i / 2;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        node.bulk_load(keys, payload, 200);

        for (int i = 0; i < 75; i++) {
            int num_erased = node.erase(i);
            EXPECT_EQ(num_erased, 2);
        }

        for (int i = 75; i < 100; i++) {
            int pos = node.find_key(i);
            EXPECT_EQ(i, node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
            node.erase(i);
        }

        EXPECT_TRUE(node.num_keys_ == 0);
    }

    TEST(DataNode, TestEraseRange)
    {
        AlexDataNode<int, int> node;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = i;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        node.bulk_load(keys, payload, 200);

        int num_erased = node.erase_range(50, 100);
        EXPECT_EQ(num_erased, 50);

        num_erased = node.erase_range(-50, 50);
        EXPECT_EQ(num_erased, 50);

        num_erased = node.erase_range(150, 300);
        EXPECT_EQ(num_erased, 50);

        for (int i = 100; i < 150; i++) {
            int pos = node.find_key(i);
            EXPECT_EQ(i, node.get_key(pos));
            EXPECT_TRUE(node.check_exists(pos));
            node.erase_range(i, i+1);
        }

        EXPECT_TRUE(node.num_keys_ == 0);
    }

    TEST(DataNode, TestBuildIndexWithSample)
    {
        const int num_keys = 20000;
        int keys[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys[i] = rand() % 50000;
        }
        std::sort(keys, keys + num_keys);

        LinearModel<int> model;
        LinearModel<int> model_using_sample;

        AlexDataNode<int, int>::build_model(keys, num_keys, &model, false);
        AlexDataNode<int, int>::build_model(keys, num_keys, &model_using_sample, true);

        double rel_diff_in_a = std::abs((model.a_ - model_using_sample.a_) / model.a_);
        double rel_diff_in_b = std::abs((model.b_ - model_using_sample.b_) / model.b_);
        EXPECT_LT(rel_diff_in_a, 0.05);
        EXPECT_LT(rel_diff_in_b, 0.05);
    }

    TEST(DataNode, TestComputeCostWithSample)
    {
        const int num_keys = 20000;
        int keys[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys[i] = rand() % 50000;
        }
        std::sort(keys, keys + num_keys);

        LinearModel<int> model;
        AlexDataNode<int, int>::build_model(keys, num_keys, &model);
        double density = 0.7;
        double expected_insert_frac = 0.5;
        ExpectedIterationsAndShiftsAccumulator ent;
        ExpectedIterationsAndShiftsAccumulator ent_using_sample;

        DataNodeStats stats;
        DataNodeStats stats_using_sample;
        AlexDataNode<int, int>::compute_expected_cost(keys, num_keys, density, expected_insert_frac, &model, false, &stats);
        AlexDataNode<int, int>::compute_expected_cost(keys, num_keys, density, expected_insert_frac, &model, true, &stats_using_sample);

        double exp_iters = stats.num_search_iterations;
        double exp_iters_sample = stats_using_sample.num_search_iterations;
        double exp_shifts = stats.num_shifts;
        double exp_shifts_sample = stats_using_sample.num_shifts;
        double rel_diff_in_search_entropy = std::abs((exp_iters - exp_iters_sample) / exp_iters);
        double rel_diff_in_shifts_entropy = std::abs((exp_shifts - exp_shifts_sample) / exp_shifts);
        EXPECT_LT(rel_diff_in_search_entropy, 0.5);
        EXPECT_LT(rel_diff_in_shifts_entropy, 0.5);
    }
};