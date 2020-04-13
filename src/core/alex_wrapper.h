// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <fstream>
#include <iostream>
#include <stack>
#include "../alex/base.h"
#include "alex.h"

// Wrapper around Alex, only used for interfacing with benchmarks
// If you want to simply use Alex, you do not need to use this wrapper
template <class T, class P, class V=std::pair<T, P>>
class AlexTrieWrapper : public LearnedIndex<T, P>
{
public:
    alex::Alex<T,P> alex_;

	// num_models is unused, you can set it to 0
    explicit AlexTrieWrapper(int num_models) : LearnedIndex<T,P>(num_models) {}

    // num_models is unused, you can set it to 0
    AlexTrieWrapper(int num_models, std::vector<double> params) : LearnedIndex<T,P>(num_models) {
        if (params.size() > 0)
            alex_.experimental_params_.fanout_selection_method = (int)params[0];
//        if (params.size() > 1)
//            alex_.m_params.root_spline = (bool)params[1];
        if (params.size() > 2)
            alex_.experimental_params_.splitting_policy_method = (int)params[2];
        if (params.size() > 3)
            alex_.set_expected_insert_frac(params[3]);
//        if (params.size() > 4)
//            alex_.params_.allow_root_expansion = (bool)params[4];
        if (params.size() > 5) {
            alex_.set_max_node_size(1 << (int)params[5]);
        }
        if (params.size() > 6) {
            if ((int)params[6] > 0) {
                alex_.set_approximate_model_computation(true);
            }
            if ((int)params[6] > 1) {
                alex_.set_approximate_cost_computation(true);
            }
        }
        if (params.size() > 7) {
            alex::kExpSearchIterationsWeight = params[7];
            alex::kShiftsWeight = params[8];
            alex::kNodeLookupsWeight = params[9];
            alex::kModelSizeWeight = params[10];
        }
    }

	~AlexTrieWrapper() = default;

	void bulk_load(T array[], P payload[], int num_keys) {
		alex_.bulk_load(array, payload, num_keys);
	}

	void insert(T key, P payload) {
        alex_.insert(key, payload);
	}

	bool erase(T key) {
		return alex_.erase(key);
	}

    inline PAYLOAD_SUM_TYPE get_key(T key)
    {
		P* payload = alex_.get_payload(key);
		if (!payload) {
			return 0;
		}
#ifdef BIG_PAYLOAD
		return payload->first_value;
#else
		return *payload;
#endif
    }

    P* get_payload_verbose(T key) {
        auto leaf = get_leaf_verbose(key);
        std::cout << "[get_payload] level " << leaf->level_
                  << " first key " << leaf->first_key() << " last key " << leaf->last_key()
                  << " num keys " << leaf->num_keys_
                  << " valid " << leaf->validate_structure()
                  << " key exists " << leaf->key_exists(key, true)
                  << std::endl;
        int idx = leaf->find_key(key);
        if (idx < 0) {
            return nullptr;
        } else {
            return &(leaf->get_payload(idx));
        }
    }

    alex::AlexDataNode<T,P>* get_leaf_verbose(T key) {
        alex::AlexNode<T,P>* cur = alex_.root_node_;

        while (!cur->is_leaf_) {
            auto node = (alex::AlexModelNode<T,P>*)cur;
            int bucketID = node->model_.predict(key);
            bucketID = std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
            cur = node->children_[bucketID];
            T node_range_start = -node->model_.b_ / node->model_.a_;
            T node_range_end = (node->num_children_ - node->model_.b_) / node->model_.a_;
            int repeats = 1 << cur->duplication_factor_;
            int start_bucketID = bucketID - (bucketID % repeats);
            int end_bucketID = start_bucketID + repeats;
            T child_range_start = (start_bucketID - node->model_.b_) / node->model_.a_;
            T child_range_end = (end_bucketID - node->model_.b_) / node->model_.a_;
            std::cout << "[get_leaf] level " << node->level_
                      << " num children " << node->num_children_
                      << " model " << node->model_.a_ << " " << node->model_.b_
                      << " node key range [" << node_range_start << ", " << node_range_end << "]"
                      << " child no " << bucketID
                      << " child duplicate range [" << start_bucketID << ", " << end_bucketID << ")"
                      << " child key range [" << child_range_start << ", " << child_range_end << "]"
                      << std::endl;
        }

        return (alex::AlexDataNode<T,P>*)cur;
    }

    PAYLOAD_SUM_TYPE get_key_verbose(T key)
    {
        P* payload = get_payload_verbose(key);
        if (!payload) {
            return 0;
        }
#ifdef BIG_PAYLOAD
        return payload->first_value;
#else
        return *payload;
#endif
    }

	inline int predict_key(T key)
	{
        alex::AlexNode<T,P>* cur = alex_.root_node_;

        while (!cur->is_leaf_) {
            auto node = (alex::AlexModelNode<T,P>*)cur;
            int bucketID = node->model_.predict(key);
            bucketID = std::min<int>(std::max<int>(bucketID, 0), node->num_children_ - 1);
            cur = node->children_[bucketID];
        }

		return ((alex::AlexDataNode<T,P>*)cur)->predict_position(key);
	}

	inline double data_size() {
        return (double)alex_.data_size();
	}

	inline long long model_size() {
        return alex_.model_size();
	}

	long long num_shifts()
	{
        long long shifts = 0;
        typename alex::Alex<T,P>::NodeIterator node_it(&alex_);
        for (; !node_it.is_end(); node_it.next()) {
            alex::AlexNode<T,P>* cur = node_it.current();
            if (cur->is_leaf_) {
                shifts += ((alex::AlexDataNode<T,P>*)cur)->num_shifts_;
            }
        }
        return shifts;
	}

    long long num_moves()
    {
        long long shifts = 0;
        typename alex::Alex<T,P>::NodeIterator node_it(&alex_);
        for (; !node_it.is_end(); node_it.next()) {
            alex::AlexNode<T,P>* cur = node_it.current();
            if (cur->is_leaf_) {
                shifts += ((alex::AlexDataNode<T,P>*)cur)->num_shifts_;
            }
        }
        return shifts;
    }

    long long num_search_iterations()
    {
        long long iters = 0;
        typename alex::Alex<T,P>::NodeIterator node_it(&alex_);
        for (; !node_it.is_end(); node_it.next()) {
            alex::AlexNode<T,P>* cur = node_it.current();
            if (cur->is_leaf_) {
                iters += ((alex::AlexDataNode<T,P>*)cur)->num_exp_search_iterations_;
            }
        }
        return iters;
    }

    int num_resizes()
    {
        int resizes = 0;
        typename alex::Alex<T,P>::NodeIterator node_it(&alex_);
        for (; !node_it.is_end(); node_it.next()) {
            alex::AlexNode<T,P>* cur = node_it.current();
            if (cur->is_leaf_) {
                resizes += ((alex::AlexDataNode<T,P>*)cur)->num_resizes_;
            }
        }
        return resizes + alex_.stats_.num_expand_and_scales;
    }

	int num_nodes() {
		return alex_.num_nodes();
	};

	int num_leaves() {
		return alex_.num_leaves();
	};

    PAYLOAD_SUM_TYPE sum_range_length(T start, int num_records_to_scan) {
        PAYLOAD_SUM_TYPE sum = 0;
        typename alex::Alex<T,P>::Iterator it = alex_.find(start);

        for (int i = 0; i < num_records_to_scan; i++) {
            if (it.is_end()) {
                break;
            }
#ifdef BIG_PAYLOAD
			sum += (*it).second.first_value;
#else
			sum += (*it).second;
#endif
            it++;
        }

        return sum;
    }
};

