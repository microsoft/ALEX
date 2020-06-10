// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gtest/gtest.h"

#define private public
#include "../src/core/alex.h"

using namespace alex;

namespace test
{

    TEST(Alex, TestBulkLoad)
    {
        Alex<int, int> index;

        int keys[500];
        int payload[500];
        for (int i = 0; i < 500; i++) {
            keys[i] = rand() % 5000;
            payload[i] = i;
        }

        std::sort(keys, keys + 500);
        index.bulk_load(keys, payload, 500);

        for (int i = 0; i < 500; i++) {
            auto it = index.find(keys[i]);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(keys[i], it.key());
        }
    }

    TEST(Alex, TestConstructors)
    {
        Alex<int, int> index;

        int keys[500];
        int payload[500];
        for (int i = 0; i < 500; i++) {
            keys[i] = rand() % 5000;
            payload[i] = i;
        }

        std::sort(keys, keys + 500);
        index.bulk_load(keys, payload, 500);

        Alex<int, int> index2(index);  // Copy constructor
        Alex<int, int> index3;
        index3 = index;  // Assignment

        EXPECT_NE(index.root_node_, index2.root_node_);
        EXPECT_NE(index.root_node_, index3.root_node_);

        for (int i = 0; i < 500; i++) {
            auto it2 = index2.find(keys[i]);
            EXPECT_TRUE(!it2.is_end());
            EXPECT_EQ(keys[i], it2.key());

            auto it3 = index3.find(keys[i]);
            EXPECT_TRUE(!it3.is_end());
            EXPECT_EQ(keys[i], it3.key());
        }
    }

    TEST(Alex, TestIterators)
    {
        Alex<int, int> index;

        int keys[500];
        int payload[500];
        for (int i = 0; i < 250; i++) {
            keys[i] = i;
            payload[i] = i;
        }
        for (int i = 250; i < 500; i++) {
            keys[i] = 5 * i;
            payload[i] = i;
        }

        std::sort(keys, keys + 500);
        index.bulk_load(keys, payload, 500);

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
        auto it = index.find(keys[250]);
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
        typename Alex<int, int>::ConstIterator cit = index.find(keys[250]);
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

    TEST(Alex, TestFind)
    {
        Alex<int, int> index;

        int keys[500];
        int payload[500];
        // even numbers from 0 to 998 inclusive
        for (int i = 0; i < 500; i++) {
            keys[i] = i * 2;
            payload[i] = i;
        }

        std::sort(keys, keys + 500);
        index.bulk_load(keys, payload, 500);

        // Find existent keys
        for (int i = 0; i < 500; i++) {
            auto it = index.find(keys[i]);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(keys[i], it.key());

            int* p = index.get_payload(keys[i]);
            EXPECT_TRUE(p);
            EXPECT_EQ(payload[i], *p);
        }

        // Find non-existent keys
        for (int i = 1; i < 100; i += 2) {
            auto it = index.find(i);
            EXPECT_TRUE(it.is_end());

            int* p = index.get_payload(i);
            EXPECT_TRUE(!p);
        }
    }

    TEST(Alex, TestLowerUpperBound)
    {
        Alex<int, int> index;

        int keys[100];
        int payload[100];
        // 10 each of 0, 10, ..., 90
        for (int i = 0; i < 100; i += 10) {
            for (int j = 0; j < 10; j++) {
                keys[i+j] = i;
                payload[i+j] = 0;
            }
        }

        std::sort(keys, keys + 100);
        index.bulk_load(keys, payload, 100);

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
        }
    }

    TEST(Alex, TestFindLastNoGreaterThan)
    {
        Alex<int, int> index;

        int keys[500];
        int payload[500];
        // even numbers from 0 to 998 inclusive
        for (int i = 0; i < 500; i++) {
            keys[i] = i * 2;
            payload[i] = i;
        }

        std::sort(keys, keys + 500);
        index.bulk_load(keys, payload, 500);

        // Existent keys
        for (int i = 0; i < 500; i++) {
            auto it = index.find_last_no_greater_than(keys[i]);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(keys[i], it.key());

            int* p = index.get_payload_last_no_greater_than(keys[i]);
            EXPECT_TRUE(p);
            EXPECT_EQ(payload[i], *p);
        }

        // Non-existent keys
        for (int i = 0; i < 500; i++) {
            int key = keys[i] + 1;
            auto it = index.find_last_no_greater_than(key);
            EXPECT_TRUE(!it.is_end());
            EXPECT_LE(it.key(), key);
            it++;
            if (!it.is_end()) {
                EXPECT_GT(it.key(), key);
            }

            int* p = index.get_payload_last_no_greater_than(key);
            EXPECT_TRUE(p);
            EXPECT_EQ(payload[i], *p);
        }

        // Non-existent key smaller than min
        auto it = index.find_last_no_greater_than(-1);
        EXPECT_TRUE(!it.is_end());
        EXPECT_EQ(keys[0], it.key());
    }

    TEST(Alex, TestReadModifyWrite)
    {
        Alex<int, int> index;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = i;
            payload[i] = 0;
        }

        std::sort(keys, keys + 100);
        index.bulk_load(keys, payload, 100);

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

    TEST(Alex, TestSequentialInserts)
    {
        Alex<int, int> index;

        int keys[50];
        int payload[50];
        for (int i = 0; i < 50; i++) {
            keys[i] = i;
            payload[i] = i;
        }

        std::sort(keys, keys + 50);
        index.bulk_load(keys, payload, 50);

        for (int i = 50; i < 200; i++) {
            index.insert(i, i);
        }

        for (int i = 0; i < 200; i++) {
            auto it = index.find(i);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(i, it.key());
        }
    }

    TEST(Alex, TestOrderedInserts)
    {
        Alex<int, int> index;

        int keys[100];
        int payload[100];
        for (int i = 0; i < 100; i++) {
            keys[i] = 2 * i;
            payload[i] = i;
        }

        std::sort(keys, keys + 100);
        index.bulk_load(keys, payload, 100);

        for (int i = 0; i < 100; i++) {
            index.insert(2 * i + 1, i);
        }

        // Check that getting the key is correct.
        for (int i = 0; i < 200; i++) {
            auto it = index.find(i);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(i, it.key());
        }
    }

    TEST(Alex, TestRandomInserts)
    {
        Alex<int, int> index;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = rand() % 500;
            payload[i] = i;
        }

        std::sort(keys, keys + 25);
        index.bulk_load(keys, payload, 25);

        for (int i = 25; i < 200; i++) {
            index.insert(keys[i], payload[i]);
        }

        // Check that getting the key is correct.
        for (int i = 0; i < 200; i++) {
            auto it = index.find(keys[i]);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(keys[i], it.key());
        }
    }

    TEST(Alex, TestInsertFromEmpty)
    {
        Alex<int, int> index;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = rand() % 500;
            payload[i] = i;
        }

        for (int i = 0; i < 200; i++) {
            index.insert(keys[i], payload[i]);
        }

        // Check that getting the key is correct.
        for (int i = 0; i < 200; i++) {
            auto it = index.find(keys[i]);
            EXPECT_TRUE(!it.is_end());
            EXPECT_EQ(keys[i], it.key());
        }
    }

    TEST(Alex, TestRandomErases)
    {
        Alex<int, int> index;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = rand() % 500;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        index.bulk_load(keys, payload, 200);

        // try to erase a nonexistent key
        EXPECT_EQ(index.erase_one(1000), 0);

        for (int i = 0; i < 200; i++) {
            int num_erased = index.erase_one(keys[i]);
            EXPECT_EQ(num_erased, 1);
        }

        EXPECT_EQ(index.stats_.num_keys, 0);
    }

    TEST(Alex, TestRangeScan)
    {
        Alex<int, int> index;

        int keys[200];
        int payload[200];
        for (int i = 0; i < 200; i++) {
            keys[i] = i;
            payload[i] = i;
        }

        std::sort(keys, keys + 200);
        index.bulk_load(keys, payload, 200);

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

    TEST(Alex, TestDebug)
    {

    }
};