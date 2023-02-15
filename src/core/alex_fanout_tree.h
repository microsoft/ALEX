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
          fanout_tree[level][2 * i].use = false;
          fanout_tree[level][2 * i + 1].use = false;
          fanout_tree[level - 1][i].use = true;
          at_least_one_merge = true;
          best_cost -= kModelSizeWeight * sizeof(AlexDataNode<T, P>) *
                       total_keys / num_keys;
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
          fanout_tree[level][2 * i].use = false;
          fanout_tree[level][2 * i + 1].use = false;
          fanout_tree[level - 1][i].use = true;
          best_cost -= merging_cost_saving * num_node_keys / num_keys;
          at_least_one_merge = true;
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
  double a[node->model_.max_key_length_] = {0.0};
  for (unsigned int i = 0; i < node->model_.max_key_length_; i++) {
    a[i] = node->model_.a_[i] * fanout;
  }
  double b = node->model_.b_ * fanout;
  LinearModel<T> newLModel(a, b, node->model_.max_key_length_);
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
        arb += a[0] * values[right_boundary].first[0] + b;
      }
      if (static_cast<int>(arb) <= i) {right_boundary++;}
    }
#if DEBUG_PRINT
    std::cout << "compute_level boundary searching finished for fanout " << fanout << std::endl;
    std::cout << "left_boundary is : " <<  left_boundary << std::endl;
    std::cout << "right_boundary is : " << right_boundary << std::endl;
#endif
    if (left_boundary == right_boundary) {
      double *slope = new double[node->max_key_length_]();
      used_fanout_tree_nodes.push_back(
          {level, i, 0, left_boundary, right_boundary, false, 0, 0, slope, 0, 0});
      continue;
    }
    LinearModel<T> model(node->model_.max_key_length_);
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
    double *slope = new double[node->model_.max_key_length_]();
    std::copy(model.a_, model.a_ + model.max_key_length_, slope);

    used_fanout_tree_nodes.push_back(
        {level, i, node_cost, left_boundary, right_boundary, false,
         stats.num_search_iterations, stats.num_shifts, slope, model.b_,
         right_boundary - left_boundary});
  }
  double traversal_cost =
      kNodeLookupsWeight +
      (kModelSizeWeight * fanout *
       (sizeof(AlexDataNode<T, P>) + sizeof(void*)) * total_keys / num_keys);
#if DEBUG_PRINT
  std::cout << "total node_cost : " << cost << ", traversal_cost : " << traversal_cost << std::endl;
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
  std::cout << "called find_best_fanout_bottom_up" << std::endl;
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
#if DEBUG_PRINT
  std::cout << "find_best_fanout_bottom_up finished" << std::endl;
#endif
  return std::make_pair(best_level, best_cost);
}

// This method is only used for experimental purposes.
// Figures out the optimal partitioning of children in a "top-down" fashion.
// Assumes node has already been trained to produce a CDF value in the range [0,
// 1).
// Returns the depth of the best fanout tree and the total cost of the fanout
// tree.
template <class T, class P>
std::pair<int, double> find_best_fanout_top_down(
    const std::pair<AlexKey<T>, P> values[], int num_keys, const AlexNode<T, P>* node,
    int total_keys, std::vector<FTNode>& used_fanout_tree_nodes, int max_fanout,
    double expected_insert_frac = 0, bool approximate_model_computation = true,
    bool approximate_cost_computation = false) {
  // Grow the fanout tree top-down breadth-first, each node independently
  // instead of complete levels at a time
  std::vector<std::vector<FTNode>> fanout_tree;
  double overall_cost = node->cost_ + kNodeLookupsWeight;
  double *slope = new double[node->max_key_length_]();
  fanout_tree.push_back({{0, 0, overall_cost, 0, num_keys, true, 0, 0, slope, 0, 0}});
  int fanout_tree_level = 1;
  int fanout = 2;
  while (true) {
    if (fanout > max_fanout) {
      // use nodes up to the previous level
      for (FTNode& tree_node : fanout_tree[fanout_tree_level - 1]) {
        tree_node.use = true;
      }
      fanout_tree_level--;
      break;
    }
    std::vector<FTNode> new_level;
    double a[node->model_.max_key_length_] = {0.0};
    for (unsigned int i = 0; i < node->model_.max_key_length_; i++) {
      a[i] = node->model_.a_[i] * fanout;
    }
    double b = node->model_.b_ * fanout;
    LinearModel<T> newLModel(a, b, node->model_.max_key_length_);
    double cost_savings_from_level = 0;

    for (FTNode& tree_node : fanout_tree[fanout_tree_level - 1]) {
      if (tree_node.left_boundary == tree_node.right_boundary) {
        continue;
      }
      int middle_boundary;
      char flag = 1;
      for (int idx = tree_node.left_boundary; idx < tree_node.right_boundary; idx++) {
        double predicted_pos = newLModel.predict_double(values[idx].first);
        if (predicted_pos >= 2 * tree_node.node_id + 1) {
          flag = 0;
          middle_boundary = idx;
          break;
        }
      }
      if (flag) {middle_boundary = tree_node.right_boundary;}

      double node_split_cost = 0;
      int num_node_keys = tree_node.right_boundary - tree_node.left_boundary;
      int boundaries[] = {tree_node.left_boundary, middle_boundary,
                          tree_node.right_boundary};
      double node_costs[2];
      DataNodeStats node_stats[2];
      LinearModel<T> node_models[2];
      node_models[0].max_key_length_ = node->model_.max_key_length_;
      node_models[1].max_key_length_ = node->model_.max_key_length_;
      for (int i = 0; i < 2; i++) {
        int left = boundaries[i];
        int right = boundaries[i + 1];
        if (left == right) {
          continue;
        }
        AlexDataNode<T, P>::build_model(values + left, right - left,
                                        &node_models[i],
                                        approximate_model_computation);
        node_costs[i] = AlexDataNode<T, P>::compute_expected_cost(
            values + left, right - left, AlexDataNode<T, P>::kInitDensity_,
            expected_insert_frac, &node_models[i], approximate_cost_computation,
            &node_stats[i]);
      }
      node_split_cost += sizeof(AlexDataNode<T, P>) * kModelSizeWeight *
                         total_keys / num_node_keys;
      if (node_split_cost < tree_node.cost) {
        cost_savings_from_level +=
            (tree_node.cost - node_split_cost) * num_node_keys / num_keys;
        for (int i = 0; i < 2; i++) {
          double *slope = new double[node_models[i].max_key_length_];
          std::copy(node_models[i].a_, node_models[i].a_ + node_models[i].max_key_length_, slope);
          new_level.push_back({fanout_tree_level, 2 * tree_node.node_id + i,
                               node_costs[i], boundaries[i], boundaries[i + 1],
                               true, node_stats[i].num_search_iterations,
                               node_stats[i].num_shifts, slope,
                               node_models[i].b_,
                               boundaries[i + 1] - boundaries[i]});
        }
        tree_node.use = false;
      }
    }
    if (new_level.empty()) {
      // use nodes up to the previous level
      fanout_tree_level--;
      break;
    }
    double level_cost = kModelSizeWeight * sizeof(void*) * fanout / 2 *
                        total_keys / num_keys;  // cost of 2X pointers
    if (level_cost > cost_savings_from_level) {
      // use nodes up to the previous level
      for (FTNode& tree_node : fanout_tree[fanout_tree_level - 1]) {
        tree_node.use = true;
      }
      fanout_tree_level--;
      break;
    }
    overall_cost -= (cost_savings_from_level - level_cost);
    fanout_tree.push_back(new_level);
    fanout_tree_level++;
    fanout *= 2;
  }
  collect_used_nodes(fanout_tree, fanout_tree_level, used_fanout_tree_nodes);
  return std::make_pair(fanout_tree_level, overall_cost);
}

/*** Method used when splitting after a node becomes full due to inserts ***/

// Figures out the optimal partitioning for the keys in an existing data node.
// Limit the maximum allowed fanout of the partitioning using max_fanout.
// This mirrors the logic of finding the best fanout "bottom-up" when bulk
// loading.
// Returns the depth of the best fanout tree.
template <class T, class P>
int find_best_fanout_existing_node(const AlexModelNode<T, P>* parent,
                                   int bucketID, int total_keys,
                                   std::vector<FTNode>& used_fanout_tree_nodes,
                                   int max_fanout) {
  // Repeatedly add levels to the fanout tree until the overall cost of each
  // level starts to increase
  auto node = static_cast<AlexDataNode<T, P>*>(parent->children_[bucketID]);
  int num_keys = node->num_keys_;
  int best_level = 0;
  double best_cost = std::numeric_limits<double>::max();
  std::vector<double> fanout_costs;
  std::vector<std::vector<FTNode>> fanout_tree;

  int repeats = 1 << node->duplication_factor_;
  int start_bucketID =
      bucketID - (bucketID % repeats);  // first bucket with same child
  int end_bucketID =
      start_bucketID + repeats;  // first bucket with different child

  T left_boundary_value[parent->max_key_length_];
  T right_boundary_value[parent->max_key_length_];
  parent->model_.find_minimum(parent->Mnode_min_key_, start_bucketID, left_boundary_value);
  parent->model_.find_minimum(left_boundary_value, end_bucketID, right_boundary_value);

  /* needs change compared to original since we now we need length-dimension realted line.
     * steps are like this
     * 1) obtain direction vector using min_key, max_key to find line going through those two.
     * 2) find t and v such that t(v(x-min_key)) = 0, and compute a_, b_ value. 
     * some linear algebra method used for finding t and v. */
  LinearModel<T> base_model(parent->model_.max_key_length_);
  double direction_vector_[parent->max_key_length_] = {0.0};
  for (unsigned int i = 0; i < base_model.max_key_length_; i++) {
    direction_vector_[i] = (double) right_boundary_value[i] - left_boundary_value[i];
  }
  for (unsigned int i = 0; i < base_model.max_key_length_; i++) {
    base_model.a_[i] = 1.0 / (direction_vector_[i] * base_model.max_key_length_);
    base_model.b_ -= left_boundary_value[i] / (direction_vector_[i]);
  }
  base_model.b_ /= base_model.max_key_length_;

  for (int fanout = 1, fanout_tree_level = 0; fanout <= max_fanout;
       fanout *= 2, fanout_tree_level++) {
    std::vector<FTNode> new_level;
    double cost = 0.0;
    double a[base_model.max_key_length_] = {0.0};
    for (unsigned int i = 0; i < base_model.max_key_length_; i++) {
      a[i] = base_model.a_[i] * fanout;
    }
    double b = base_model.b_ * fanout;
    LinearModel<T> newLModel(a, b, base_model.max_key_length_);
    int left_boundary = 0;
    int right_boundary = 0;
    for (int i = 0; i < fanout; i++) {
      left_boundary = right_boundary;
      if (i == fanout - 1) {right_boundary = node->data_capacity_;}
      else {
        newLModel.find_minimum(left_boundary_value, i + 1, left_boundary_value);
        AlexKey<T> pos_find_key = AlexKey<T>(left_boundary_value, node->max_key_length_);
        right_boundary = node->lower_bound(pos_find_key);
      }
      if (left_boundary == right_boundary) {
        double *slope = new double[parent->max_key_length_]();
        new_level.push_back({fanout_tree_level, i, 0, left_boundary,
                             right_boundary, false, 0, 0, slope, 0, 0});
        continue;
      }
      int num_actual_keys = 0;
      LinearModel<T> model(node->model_.max_key_length_);
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
  merge_nodes_upwards<T, P>(best_level, best_cost, num_keys, total_keys,
                            fanout_tree);

  collect_used_nodes(fanout_tree, best_level, used_fanout_tree_nodes);
  return best_level;
}

}  // namespace fanout_tree

}  // namespace alex
