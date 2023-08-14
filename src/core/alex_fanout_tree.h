// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This file contains utility code for using the fanout tree to help ALEX
 * decide the best fanout and key partitioning scheme for ALEX nodes
 * during bulk loading and node splitting.
 */

#pragma once

#include "alex_base.h"
#include "alex_nodes.h"

namespace alex {

namespace fanout_tree {

// A node of the fanout tree
struct FTNode {
  int level;    // level in the fanout tree
  int node_id;  // node's position within its level
  double cost;
  int left_boundary;  // start position in input array that this node represents
  int right_boundary;  // end position (exclusive) in input array that this node
                       // represents
  bool use = false;
  double expected_avg_search_iterations = 0;
  double expected_avg_shifts = 0;
  double *a = nullptr;  // linear model slope
  double b = 0;  // linear model intercept
  int num_keys = 0;
};

/*** Helpers ***/

// Collect all used fanout tree nodes and sort them
inline void collect_used_nodes(const std::vector<std::vector<FTNode>>& fanout_tree,
                        int max_level,
                        std::vector<FTNode>& used_fanout_tree_nodes) {
  max_level = std::min(max_level, static_cast<int>(fanout_tree.size()) - 1);
  for (int i = 0; i <= max_level; i++) {
    auto& level = fanout_tree[i];
    for (const FTNode& tree_node : level) {
      if (tree_node.use) {
        used_fanout_tree_nodes.push_back(tree_node);
      }
    }
  }
  std::sort(used_fanout_tree_nodes.begin(), used_fanout_tree_nodes.end(),
            [&](FTNode& left, FTNode& right) {
              // this is better than comparing boundary locations
              return (left.node_id << (max_level - left.level)) <
                     (right.node_id << (max_level - right.level));
            });
}

// Starting from a complete fanout tree of a certain depth, merge tree nodes
// upwards if doing so decreases the cost.
// Returns the new best cost.
// This is a helper function for finding the best fanout in a bottom-up fashion.
template <class T, class P>
static double merge_nodes_upwards(
    int start_level, double best_cost, int num_keys, int total_keys,
    std::vector<std::vector<FTNode>>& fanout_tree) {
  for (int level = start_level; level >= 1; level--) {
    int level_fanout = 1 << level;
    bool at_least_one_merge = false;
    for (int i = 0; i < level_fanout / 2; i++) {
      if (fanout_tree[level][2 * i].use && fanout_tree[level][2 * i + 1].use) {
        int num_node_keys = fanout_tree[level - 1][i].num_keys;
        if (num_node_keys == 0) {
          if (fanout_tree[level][2*i].left_boundary != fanout_tree[level-1][i].left_boundary
            ||fanout_tree[level][2*i+1].right_boundary != fanout_tree[level-1][i].right_boundary)
          {continue;} //shouldn't happen. semantic issue.
          fanout_tree[level][2 * i].use = false;
          fanout_tree[level][2 * i + 1].use = false;
          fanout_tree[level - 1][i].use = true;
          at_least_one_merge = true;
          best_cost -= kModelSizeWeight * sizeof(AlexDataNode<T, P>) *
                       total_keys / num_keys;
#if DEBUG_PRINT
          //std::cout << "merging\n where first l/r boundary is " << fanout_tree[level][2 * i].left_boundary
          //          << " and " << fanout_tree[level][2 * i].right_boundary << '\n'
          //          << "and second l/r boundary is " << fanout_tree[level][2 * i + 1].left_boundary
          //          << " and " << fanout_tree[level][2 * i + 1].right_boundary << '\n'
          //          << "which will use following boundary : " << fanout_tree[level - 1][i].left_boundary
          //          << ", " << fanout_tree[level - 1][i].right_boundary << '\n';
#endif
          continue;
        }
        int num_left_keys = fanout_tree[level][2 * i].num_keys;
        int num_right_keys = fanout_tree[level][2 * i + 1].num_keys;
        double merging_cost_saving =
            (fanout_tree[level][2 * i].cost * num_left_keys / num_node_keys) +
            (fanout_tree[level][2 * i + 1].cost * num_right_keys /
             num_node_keys) -
            fanout_tree[level - 1][i].cost +
            (kModelSizeWeight * sizeof(AlexDataNode<T, P>) * total_keys /
             num_node_keys);
        if (merging_cost_saving >= 0) {
          if (fanout_tree[level][2*i].left_boundary != fanout_tree[level-1][i].left_boundary
            ||fanout_tree[level][2*i+1].right_boundary != fanout_tree[level-1][i].right_boundary)
          {continue;} //shouldn't happen. semantic issue.
          fanout_tree[level][2 * i].use = false;
          fanout_tree[level][2 * i + 1].use = false;
          fanout_tree[level - 1][i].use = true;
          best_cost -= merging_cost_saving * num_node_keys / num_keys;
          at_least_one_merge = true;
#if DEBUG_PRINT
          //std::cout << "merging\n where first l/r boundary is " << fanout_tree[level][2 * i].left_boundary
          //          << " and " << fanout_tree[level][2 * i].right_boundary << '\n'
          //          << "and second l/r boundary is " << fanout_tree[level][2 * i + 1].left_boundary
          //          << " and " << fanout_tree[level][2 * i + 1].right_boundary << '\n'
          //          << "which will use following boundary : " << fanout_tree[level - 1][i].left_boundary
          //          << ", " << fanout_tree[level - 1][i].right_boundary << '\n';
#endif
        }
      }
    }
    if (!at_least_one_merge) {
      break;
    }
  }
  return best_cost;
}

/*** Methods used when bulk loading ***/

// Computes one complete level of the fanout tree.
// For example, level 3 will have 8 tree nodes, which are returned through
// used_fanout_tree_nodes.
// Assumes node has already been trained to produce a CDF value in the range [0,
// 1).
template <class T, class P>
double compute_level(const std::pair<AlexKey<T>, P> values[], int num_keys,
                     const AlexNode<T, P>* node, int total_keys,
                     std::vector<FTNode>& used_fanout_tree_nodes, int level,
                     int max_data_node_keys, double expected_insert_frac = 0,
                     bool approximate_model_computation = true,
                     bool approximate_cost_computation = false) {
  int fanout = 1 << level;
  double cost = 0.0;
  double a[node->max_key_length_] = {0.0};
  double b = 0.0;

  //for string key, we need to obtain model by retraining, not CDF [0,1]
  //I THINK THIS MEANS, THAT WE MAY DON'T NEED [0,1] CDF TRAINING IN FIRST PLACE. CONSIDER IT.
  LinearModel<T> tmp_model(node->max_key_length_);
  LinearModelBuilder<T> tmp_model_builder(&tmp_model);
  for (int i = 0; i < num_keys; i++) {
    tmp_model_builder.add(values[i].first, ((double) i / (num_keys-1)) * fanout);
  }
  tmp_model_builder.build();
  for (unsigned int i = 0; i < node->max_key_length_; i++) {
    a[i] = tmp_model.a_[i];
  }
  b = tmp_model.b_; 


  LinearModel<T> newLModel(a, b, node->max_key_length_);
  int left_boundary = 0;
  int right_boundary = 0;
#if DEBUG_PRINT
  std::cout << "compute_level searching for boundary with fanout : " << fanout << std::endl;
#endif
  for (int i = 0; i < fanout; i++) {
    left_boundary = right_boundary;
    /* some important change is made about right_boundary.
     * We are trying to obtain the first AlexKey where prediction position results as 'i',
     * considering the current model. Since we can't pinpoint the specific key for strings
     * and since values are all sorted (upon research) when called, we try to find the first key
     * that is equal or larger than position i. */
    if (i == fanout - 1) {right_boundary = num_keys;}
    else {
      char flag = 1;
      for (int idx = left_boundary; idx < num_keys; idx++) {
        double predicted_pos = newLModel.predict_double(values[idx].first);
        if (predicted_pos >= (double) i+1) {
          flag = 0;
          right_boundary = idx;
          break;
        }
      }
      if (flag) {right_boundary = num_keys;}
    }
    
    // Account for off-by-one errors due to floating-point precision issues.
    while (right_boundary < num_keys) {
      double arb = 0.0;
      for (unsigned int i = 0; i < newLModel.max_key_length_; i++) {
        arb += a[0] * values[right_boundary].first.key_arr_[0] + b;
      }
      if (static_cast<int>(arb) <= i) {right_boundary++;}
      else {break;}
    }
#if DEBUG_PRINT
    //if (left_boundary == num_keys) {
    //  std::cout << "left_boundary overflow with " << left_boundary << '\n';
    //}
    //else {
    //  std::cout << "left_boundary is : " <<  left_boundary << '\n';
    //  std::cout << "it's key is : " << values[left_boundary].first.key_arr_ << '\n'; 
    //}
    //if (right_boundary == num_keys) {
    //  std::cout << "right_boundary overflow with " << right_boundary << '\n';
    //}
    //else {
    //  std::cout << "just before right boundary is : " << right_boundary - 1 << '\n';
    //  std::cout << "it's key is : " << values[right_boundary-1].first.key_arr_ << '\n';
    //}
#endif
    if (left_boundary == right_boundary) {
      double *slope = new double[node->max_key_length_]();
      used_fanout_tree_nodes.push_back(
          {level, i, 0, left_boundary, right_boundary, false, 0, 0, slope, 0, 0});
      continue;
    }
    LinearModel<T> model(node->max_key_length_);
    AlexDataNode<T, P>::build_model(values + left_boundary,
                                    right_boundary - left_boundary, &model,
                                    approximate_model_computation);

    DataNodeStats stats;
    double node_cost = AlexDataNode<T, P>::compute_expected_cost(
        values + left_boundary, right_boundary - left_boundary,
        AlexDataNode<T, P>::kInitDensity_, expected_insert_frac, &model,
        approximate_cost_computation, &stats);
    // If the node is too big to be a data node, proactively incorporate an
    // extra tree traversal level into the cost.
    if (right_boundary - left_boundary > max_data_node_keys) {
      node_cost += kNodeLookupsWeight;
    }

    cost += node_cost * (right_boundary - left_boundary) / num_keys;
    double *slope = new double[node->max_key_length_]();
    std::copy(model.a_, model.a_ + model.max_key_length_, slope);

    used_fanout_tree_nodes.push_back(
        {level, i, node_cost, left_boundary, right_boundary, false,
         stats.num_search_iterations, stats.num_shifts, slope, model.b_,
         right_boundary - left_boundary});
  }
#if DEBUG_PRINT
    //std::cout << "compute_level boundary searching finished for fanout " << fanout << '\n';
#endif
  double traversal_cost =
      kNodeLookupsWeight +
      (kModelSizeWeight * fanout *
       (sizeof(AlexDataNode<T, P>) + sizeof(void*)) * total_keys / num_keys);
#if DEBUG_PRINT
  //std::cout << "total node_cost : " << cost << ", traversal_cost : " << traversal_cost << std::endl;
#endif
  cost += traversal_cost;
  return cost;
}

// Figures out the optimal partitioning of children in a "bottom-up" fashion
// (see paper for details).
// Assumes node has already been trained to produce a CDF value in the range [0,
// 1).
// Returns the depth of the best fanout tree and the total cost of the fanout
// tree.
template <class T, class P>
std::pair<int, double> find_best_fanout_bottom_up(
    const std::pair<AlexKey<T>, P> values[], int num_keys, const AlexNode<T, P>* node,
    int total_keys, std::vector<FTNode>& used_fanout_tree_nodes, int max_fanout,
    int max_data_node_keys, double expected_insert_frac = 0,
    bool approximate_model_computation = true,
    bool approximate_cost_computation = false) {
  // Repeatedly add levels to the fanout tree until the overall cost of each
  // level starts to increase
#if DEBUG_PRINT
  //std::cout << "called find_best_fanout_bottom_up" << std::endl;
#endif
  int best_level = 0;
  double best_cost = node->cost_ + kNodeLookupsWeight;
  std::vector<double> fanout_costs;
  std::vector<std::vector<FTNode>> fanout_tree;
  fanout_costs.push_back(best_cost);
  double *slope = new double[node->max_key_length_]();
  fanout_tree.push_back(
      {{0, 0, best_cost, 0, num_keys, false, 0, 0, slope, 0, num_keys}});
  for (int fanout = 2, fanout_tree_level = 1; fanout <= max_fanout;
       fanout *= 2, fanout_tree_level++) {
    std::vector<FTNode> new_level;
    double cost = compute_level<T, P>(
        values, num_keys, node, total_keys, new_level, fanout_tree_level,
        max_data_node_keys, expected_insert_frac, approximate_model_computation,
        approximate_cost_computation);
    fanout_costs.push_back(cost);
    if (fanout_costs.size() >= 3 &&
        fanout_costs[fanout_costs.size() - 1] >
            fanout_costs[fanout_costs.size() - 2] &&
        fanout_costs[fanout_costs.size() - 2] >
            fanout_costs[fanout_costs.size() - 3]) {
      for (const FTNode& tree_node : new_level) {delete[] tree_node.a;}
      break;
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_level = fanout_tree_level;
    }
    fanout_tree.push_back(new_level);
  }
  for (FTNode& tree_node : fanout_tree[best_level]) {
    tree_node.use = true;
  }

  // Merge nodes to improve cost
  best_cost = merge_nodes_upwards<T, P>(best_level, best_cost, num_keys,
                                        total_keys, fanout_tree);

  collect_used_nodes(fanout_tree, best_level, used_fanout_tree_nodes);

  for (const std::vector<FTNode>& level_fanout_tree : fanout_tree) {
    for (const FTNode& tree_node : level_fanout_tree) {
      if (!tree_node.use) {delete[] tree_node.a;}
    }
  }
#if DEBUG_PRINT
  //std::cout << "find_best_fanout_bottom_up finished" << std::endl;
#endif
  return std::make_pair(best_level, best_cost);
}

/*** Method used when splitting after a node becomes full due to inserts ***/

// Figures out the optimal partitioning for the keys in an existing data node.
// Limit the maximum allowed fanout of the partitioning using max_fanout.
// This mirrors the logic of finding the best fanout "bottom-up" when bulk
// loading.
// Returns the depth of the best fanout tree.
template <class T, class P>
int find_best_fanout_existing_node(AlexDataNode<T, P>* node, int total_keys,
                                   std::vector<FTNode>& used_fanout_tree_nodes, int max_fanout,
                                   uint32_t worker_id) {
  // Repeatedly add levels to the fanout tree until the overall cost of each
  // level starts to increase
  int num_keys = node->num_keys_;
  int best_level = 0;
  double best_cost = std::numeric_limits<double>::max();
  std::vector<double> fanout_costs;
  std::vector<std::vector<FTNode>> fanout_tree;
  
  LinearModel<T> base_model(node->max_key_length_);

  for (int fanout = 1, fanout_tree_level = 0; fanout <= max_fanout;
       fanout *= 2, fanout_tree_level++) {
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << " - ";
    //std::cout << "find_best_fanout_existing_node searching for boundary with fanout : " << fanout << std::endl;
    //alex::coutLock.unlock();
#endif
    std::vector<FTNode> new_level;
    double cost = 0.0;
    double a[base_model.max_key_length_] = {0.0};
    double b = 0.0;


    /* For each fanout we make corresponding model for string keys
     * by doing model building. Process is similar to bulk_load_node. */
    typedef typename AlexDataNode<T, P>::iterator_type iterator;
    LinearModel<T> tmp_model(base_model.max_key_length_);
    LinearModelBuilder<T> tmp_model_builder(&tmp_model);
    iterator it(node, 0);
      
    int key_cnt = 0;
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << " - ";
    //std::cout << "note that node's key count is : " << num_keys << std::endl;
    //alex::coutLock.unlock();
#endif

    while (it.cur_idx_ != -1) {
      tmp_model_builder.add(it.key(), ((double) key_cnt * fanout / (node->num_keys_ - 1)));
      key_cnt++;
      it++;
#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << " - ";
    //std::cout << "key_cnt is " << key_cnt << '\n';
    //std::cout << "next it idx is " << it.cur_idx_ << std::endl;
    //alex::coutLock.unlock();
#endif
    }
    tmp_model_builder.build();
    std::copy(tmp_model.a_, tmp_model.a_ + tmp_model.max_key_length_, a);
    b = tmp_model.b_;

    LinearModel<T> newLModel(a, b, base_model.max_key_length_);
#if DEBUG_PRINT
      //alex::coutLock.lock();
      //std::cout << "t" << worker_id << " - ";
      //std::cout << "first key predicted as" << newLModel.predict_double(node->key_slots_[node->first_pos()]) << '\n';
      //std::cout << "last key predicted as" << newLModel.predict_double(node->key_slots_[node->last_pos()]) << std::endl;
      //alex::coutLock.unlock();
#endif

    int left_boundary = 0;
    int right_boundary = 0;
    for (int i = 0; i < fanout; i++) {
      left_boundary = right_boundary;
      if (i == fanout - 1) {right_boundary = node->data_capacity_;}
      else {
        if (typeid(char) != typeid(T)) { /* numeric key */
          AlexKey<T> tmpkey = AlexKey<T>(node->max_key_length_);
          tmpkey.key_arr_[0] = (T) ((i + 1) - b) / a[0];
          right_boundary = node->lower_bound(tmpkey);
        }
        else { /* string key */
          /* we iterate through the key array to find the smallest key resulting to i + 1*/
          char flag = 1;
          AlexKey<T> tmpkey = AlexKey<T>(node->max_key_length_);
          for (int key_arr_idx = left_boundary; key_arr_idx < node->data_capacity_; key_arr_idx++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
            tmpkey = node->key_slots_[key_arr_idx];
#else
            tmpkey = node->data_slots_[key_arr_idx].first;
#endif
            if (node->key_equal(tmpkey, node->kEndSentinel_)) {
              continue;
            }
            else if (node->model_.predict(tmpkey) >= i+1) {
              flag = 0;
              right_boundary = key_arr_idx;
              break;
            }
          }
          if (flag) {right_boundary = node->data_capacity_ - 1;}
        }
      }

      if (left_boundary == right_boundary) {
        double *slope = new double[node->max_key_length_]();
        new_level.push_back({fanout_tree_level, i, 0, left_boundary,
                             right_boundary, false, 0, 0, slope, 0, 0});
        continue;
      }
      int num_actual_keys = 0;
      LinearModel<T> model(node->max_key_length_);
      typename AlexDataNode<T, P>::const_iterator_type it(node, left_boundary);
      LinearModelBuilder<T> builder(&model);
      for (int j = 0; it.cur_idx_ < right_boundary && !it.is_end(); it++, j++) {
        builder.add(it.key(), j);
        num_actual_keys++;
      }
      builder.build();

      double empirical_insert_frac = node->frac_inserts();
      DataNodeStats stats;
      double node_cost =
          AlexDataNode<T, P>::compute_expected_cost_from_existing(
              node, left_boundary, right_boundary,
              AlexDataNode<T, P>::kInitDensity_, empirical_insert_frac, &model,
              &stats);

      cost += node_cost * num_actual_keys / num_keys;

      double *slope = new double[model.max_key_length_];
      std::copy(model.a_, model.a_ + model.max_key_length_, slope);
      new_level.push_back({fanout_tree_level, i, node_cost, left_boundary,
                           right_boundary, false, stats.num_search_iterations,
                           stats.num_shifts, slope, model.b_,
                           num_actual_keys});
#if DEBUG_PRINT
      //alex::coutLock.lock();
      //std::cout << "t" << worker_id << " - ";
      //std::cout << "left boundary is : " <<  left_boundary << '\n';
      //std::cout << "right boundary is : " << right_boundary << std::endl;
      //alex::coutLock.unlock();
#endif
    }
    // model weight reflects that it has global effect, not local effect
    double traversal_cost =
        kNodeLookupsWeight +
        (kModelSizeWeight * fanout *
         (sizeof(AlexDataNode<T, P>) + sizeof(void*)) * total_keys / num_keys);
    cost += traversal_cost;
    fanout_costs.push_back(cost);
    // stop after expanding fanout increases cost twice in a row
    if (fanout_costs.size() >= 3 &&
        fanout_costs[fanout_costs.size() - 1] >
            fanout_costs[fanout_costs.size() - 2] &&
        fanout_costs[fanout_costs.size() - 2] >
            fanout_costs[fanout_costs.size() - 3]) {
      for (const FTNode& tree_node : new_level) {delete[] tree_node.a;}
      break;
    }
    if (cost < best_cost) {
      best_cost = cost;
      best_level = fanout_tree_level;
    }
    fanout_tree.push_back(new_level);

#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << " - ";
    //std::cout << "find_best_fanout_existing_node boundary searching finished for fanout " << fanout << std::endl;
    //alex::coutLock.unlock();
#endif
  }

#if DEBUG_PRINT
    //alex::coutLock.lock();
    //std::cout << "t" << worker_id << " - ";
    //std::cout << "chosen best level is : " << (1 << best_level) << std::endl;
    //alex::coutLock.unlock();
#endif

  for (FTNode& tree_node : fanout_tree[best_level]) {
    tree_node.use = true;
  }

  for (const std::vector<FTNode>& level_fanout_tree : fanout_tree) {
    for (const FTNode& tree_node : level_fanout_tree) {
      if (!tree_node.use) {delete[] tree_node.a;}
    }
  }

  // Merge nodes to improve cost
  merge_nodes_upwards<T, P>(best_level, best_cost, num_keys, total_keys,
                            fanout_tree);

  collect_used_nodes(fanout_tree, best_level, used_fanout_tree_nodes);
  return best_level;
}

}  // namespace fanout_tree

}  // namespace alex
