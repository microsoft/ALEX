// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * ALEX with key type T and payload type P, combined type V=std::pair<T, P>.
 * Iterating through keys is done using an "Iterator".
 * Iterating through tree nodes is done using a "NodeIterator".
 *
 * Core user-facing API of Alex:
 * - Alex()
 * - void bulk_load(V values[], int num_keys)
 * - void insert(T key, P payload)
 * - Iterator find(T key)  // for exact match
 * - Iterator begin()
 * - Iterator end()
 * - Iterator lower_bound(T key)
 * - Iterator upper_bound(T key)
 *
 * User-facing API of Iterator:
 * - void operator ++ ()  // post increment
 * - V operator * ()  // does not return reference to V by default
 * - const T& key ()
 * - P& payload ()
 * - bool is_end()
 * - bool operator == (const Iterator & rhs)
 * - bool operator != (const Iterator & rhs)
 */

#pragma once

#include <fstream>
#include <iostream>
#include <stack>
#include <type_traits>
#include <iomanip> //only for printing some debugging message.

#include "alex_base.h"
#include "alex_fanout_tree.h"
#include "alex_nodes.h"

// Whether we account for floating-point precision issues when traversing down
// ALEX.
// These issues rarely occur in practice but can cause incorrect behavior.
// Turning this on will cause slight performance overhead due to extra
// computation and possibly accessing two data nodes to perform a lookup.
#define ALEX_SAFE_LOOKUP 1

namespace alex {

template <class T, class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<AlexKey<T>, P>>,
          bool allow_duplicates = true>
class Alex {
  static_assert(std::is_arithmetic<T>::value, "ALEX key type must be numeric.");
  static_assert(std::is_same<Compare, AlexCompare>::value,
                "Must use AlexCompare.");

 public:
  // Value type, returned by dereferencing an iterator
  typedef std::pair<AlexKey<T>, P> V;

  // ALEX class aliases
  typedef Alex<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef AlexNode<T, P, Alloc> node_type;
  typedef AlexModelNode<T, P, Alloc> model_node_type;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> data_node_type;

  // Forward declaration for iterators
  class Iterator;
  class ConstIterator;
  class ReverseIterator;
  class ConstReverseIterator;
  class NodeIterator;  // Iterates through all nodes with pre-order traversal

  node_type* root_node_ = nullptr;
  model_node_type* superroot_ =
      nullptr;  // phantom node that is the root's parent
  unsigned int max_key_length_ = 1; // maximum length of keys in this ALEX structure.

  /* User-changeable parameters */
  struct Params {
    // When bulk loading, Alex can use provided knowledge of the expected
    // fraction of operations that will be inserts
    // For simplicity, operations are either point lookups ("reads") or inserts
    // ("writes)
    // i.e., 0 means we expect a read-only workload, 1 means write-only
    double expected_insert_frac = 1;
    // Maximum node size, in bytes. By default, 16MB.
    // Higher values result in better average throughput, but worse tail/max
    // insert latency
    int max_node_size = 1 << 24;
    // Approximate model computation: bulk load faster by using sampling to
    // train models
    bool approximate_model_computation = true;
    // Approximate cost computation: bulk load faster by using sampling to
    // compute cost
    bool approximate_cost_computation = false;
  };
  Params params_;

  /* Setting max node size automatically changes these parameters */
  struct DerivedParams {
    // The defaults here assume the default max node size of 16MB
    int max_fanout = 1 << 21;  // assumes 8-byte pointers
    int max_data_node_slots = (1 << 24) / sizeof(V);
  };
  DerivedParams derived_params_;

  /* Counters, useful for benchmarking and profiling */
  struct Stats {
    AtomicVal<int> num_keys = 0;
    AtomicVal<int> num_model_nodes = 0;  // num model nodes
    AtomicVal<int> num_data_nodes = 0;   // num data nodes
    AtomicVal<int> num_expand_and_scales = 0;
    AtomicVal<int> num_expand_and_retrains = 0;
    AtomicVal<int> num_downward_splits = 0;
    AtomicVal<int> num_sideways_splits = 0;
    AtomicVal<int> num_model_node_expansions = 0;
    AtomicVal<int> num_model_node_splits = 0;
    AtomicVal<long long> num_downward_split_keys = 0;
    AtomicVal<long long> num_sideways_split_keys = 0;
    AtomicVal<long long> num_model_node_expansion_pointers = 0;
    AtomicVal<long long> num_model_node_split_pointers = 0;
    AtomicVal<long long> num_node_lookups = 0;
    AtomicVal<long long> num_lookups = 0;
    AtomicVal<long long> num_inserts = 0;
    AtomicVal<double> splitting_time = 0;
    AtomicVal<double> cost_computation_time = 0;
  };
  Stats stats_;

 private:
  /* Structs used internally */
  /* Statistics related to the key domain.
   * The index can hold keys outside the domain, but lookups/inserts on those
   * keys will be inefficient.
   * If enough keys fall outside the key domain, then we expand the key domain.
   */
  struct InternalStats {
    T *key_domain_min_ = nullptr; // we need to initialize this for every initializer
    T *key_domain_max_ = nullptr; // we need to initialize this for every initializer
  };
  InternalStats istats_;

  /* Used when finding the best way to propagate up the RMI when splitting
   * upwards.
   * Cost is in terms of additional model size created through splitting
   * upwards, measured in units of pointers.
   * One instance of this struct is created for each node on the traversal path.
   * User should take into account the cost of metadata for new model nodes
   * (base_cost). */
  struct SplitDecisionCosts {
    static constexpr double base_cost =
        static_cast<double>(sizeof(model_node_type)) / sizeof(void*);
    // Additional cost due to this node if propagation stops at this node.
    // Equal to 0 if redundant slot exists, otherwise number of new pointers due
    // to node expansion.
    double stop_cost = 0;
    // Additional cost due to this node if propagation continues past this node.
    // Equal to number of new pointers due to node splitting, plus size of
    // metadata of new model node.
    double split_cost = 0;
  };

  // At least this many keys must be outside the domain before a domain
  // expansion is triggered.
  static const int kMinOutOfDomainKeys = 5;
  // After this many keys are outside the domain, a domain expansion must be
  // triggered.
  static const int kMaxOutOfDomainKeys = 1000;
  // When the number of max out-of-domain (OOD) keys is between the min and
  // max, expand the domain if the number of OOD keys is greater than the
  // expected number of OOD due to randomness by greater than the tolereance
  // factor.
  static const int kOutOfDomainToleranceFactor = 2;

  Compare key_less_ = Compare();
  Alloc allocator_ = Alloc();

  /*** Constructors and setters ***/

 public:
 /* basic initialization can handle up to 4 parameters
  * 1) max key length of each keys. default value is 1. 
  * 3) compare function used for comparing. Default is basic AlexCompare
  * 4) allocation function used for allocation. Default is basic allocator. */
  Alex() {
    // key_domain setup
    istats_.key_domain_min_ = new T[1];
    istats_.key_domain_min_[0] = STR_VAL_MAX;
    istats_.key_domain_max_ = new T[1];
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    
    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(1, nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    stats_.num_data_nodes++;
    create_superroot();
  }

  Alex(unsigned int max_key_length,
       const Compare& comp = Compare(), const Alloc& alloc = Alloc())
    : max_key_length_(max_key_length), key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length];
    istats_.key_domain_max_ = new T[max_key_length];
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length,
        STR_VAL_MAX);
    istats_.key_domain_max_[0] = STR_VAL_MIN;
    
    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(max_key_length_, nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    stats_.num_data_nodes.increment();
    create_superroot();
  }  

  Alex(const Compare& comp, const Alloc& alloc = Alloc())
      : key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[1];
    istats_.key_domain_min_[0] = STR_VAL_MAX;
    istats_.key_domain_max_ = new T[1];
    istats_.key_domain_max_[0] = STR_VAL_MIN;

    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(1, nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    stats_.num_data_nodes++;
    create_superroot();
  }

  Alex(const Alloc& alloc) : allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[1];
    istats_.key_domain_min_[0] = STR_VAL_MAX;
    istats_.key_domain_max_ = new T[1];
    istats_.key_domain_max_[0] = STR_VAL_MIN;

    // Set up root as empty data node
    auto empty_data_node = new (data_node_allocator().allocate(1))
        data_node_type(1, nullptr, key_less_, allocator_);
    empty_data_node->bulk_load(nullptr, 0);
    root_node_ = empty_data_node;
    stats_.num_data_nodes++;
    create_superroot();
  }

  //NOTE : destruction should be done when multithreading
  ~Alex() {
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      delete_node(node_it.current());
    }
    delete_node(superroot_);
    delete[] istats_.key_domain_min_;
    delete[] istats_.key_domain_max_;
  }

  // Below 4 constructors initializes with range [first, last). 
  // The range does not need to be sorted. 
  // This creates a temporary copy of the data. 
  // If possible, we recommend directly using bulk_load() instead.
  // NEED FIX (max_key_length issue, not urgent
  //           possible but not implemented since it's not used yet.)
  template <class InputIterator>
  explicit Alex(InputIterator first, InputIterator last,
                unsigned int max_key_length,
                const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : max_key_length_(max_key_length),
        key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[max_key_length];
    std::fill(istats_.key_domain_min_, istats_.key_domain_min_ + max_key_length,
              STR_VAL_MAX);
    istats_.key_domain_max_ = new T[max_key_length];
    istats_.key_domain_max_[0] = STR_VAL_MIN;

    std::vector<V> values;
    for (auto it = first; it != last; ++it) {
      values.push_back(*it);
    }
    std::sort(values.begin(), values.end(),
            [this](auto const& a, auto const& b) {return a.first < b.first;});
    bulk_load(values.data(), static_cast<int>(values.size()));
  }

  template <class InputIterator>
  explicit Alex(InputIterator first, InputIterator last,
                const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : key_less_(comp), allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[1];
    istats_.key_domain_min_[0] = STR_VAL_MAX;
    istats_.key_domain_max_ = new T[1];
    istats_.key_domain_max_[0] = STR_VAL_MIN;

    std::vector<V> values;
    for (auto it = first; it != last; ++it) {
      values.push_back(*it);
    }
    std::sort(values.begin(), values.end(),
            [this](auto const& a, auto const& b) {return a.first < b.first;});
    bulk_load(values.data(), static_cast<int>(values.size()));
  }

  template <class InputIterator>
  explicit Alex(InputIterator first, InputIterator last,
                const Alloc& alloc = Alloc())
      : allocator_(alloc) {
    // key_domain setup
    istats_.key_domain_min_ = new T[1];
    istats_.key_domain_min_[0] = STR_VAL_MAX;
    istats_.key_domain_max_ = new T[1];
    istats_.key_domain_max_[0] = STR_VAL_MIN;

    std::vector<V> values;
    for (auto it = first; it != last; ++it) {
      values.push_back(*it);
    }
    std::sort(values.begin(), values.end(),
            [this](auto const& a, auto const& b) {return a.first < b.first;});
    bulk_load(values.data(), static_cast<int>(values.size()));
  }

  //IF YOUT WANT TO USE BELOW THREE FUNCTIONS IN MULTITHREAD ALEX,
  //PLEASE CHECK IF NO THREAD IS OPERAING FOR ALEX THAT'S BEING COPIED.
  explicit Alex(const self_type& other)
      : params_(other.params_),
        derived_params_(other.derived_params_),
        stats_(other.stats_),
        istats_(other.istats_),
        key_less_(other.key_less_),
        allocator_(other.allocator_),
        max_key_length_(other.max_key_length_) {
    istats_.key_domain_min_ = new T[max_key_length_];
    istats_.key_domain_max_ = new T[max_key_length_];
    std::copy(other.istats_.key_domain_min_, other.istats_.key_domain_min_ + other.max_key_length_,
        istats_.key_domain_min_);
    std::copy(other.istats_.key_domain_max_, other.istats_.key_domain_max_ + other.max_key_length_,
        istats_.key_domain_max_);
    superroot_ =
        static_cast<model_node_type*>(copy_tree_recursive(other.superroot_));
    root_node_ = superroot_->children_[0];
  }

  Alex& operator=(const self_type& other) {
    if (this != &other) {
      for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
           node_it.next()) {
        delete_node(node_it.current());
      }
      delete_node(superroot_);
      delete[] istats_.key_domain_min_;
      delete[] istats_.key_domain_max_;
      params_ = other.params_;
      derived_params_ = other.derived_params_;
      istats_ = other.istats_;
      stats_ = other.stats_;
      key_less_ = other.key_less_;
      allocator_ = other.allocator_;
      max_key_length_ = other.max_key_length_;
      istats_.key_domain_min_ = new T[other.max_key_length_];
      istats_.key_domain_max_ = new T[other.max_key_length_];
      std::copy(other.istats_.key_domain_min_, other.istats_.key_domain_min_ + other.max_key_length_,
          istats_.key_domain_min_);
      std::copy(other.istats_.key_domain_max_, other.istats_.key_domain_max_ + other.max_key_length_,
          istats_.key_domain_max_);
      superroot_ =
          static_cast<model_node_type*>(copy_tree_recursive(other.superroot_));
      root_node_ = superroot_->children_[0];
    }
    return *this;
  }

  void swap(const self_type& other) {
    std::swap(params_, other.params_);
    std::swap(derived_params_, other.derived_params_);
    std::swap(istats_, other.istats_);
    std::swap(stats_, other.stats_);
    std::swap(key_less_, other.key_less_);
    std::swap(allocator_, other.allocator_);

    unsigned int arb_max_key_length_ = max_key_length_;
    max_key_length_ = other.max_key_length_;
    other.max_key_length_ = arb_max_key_length_;

    std::swap(istats_.key_domain_min_, other.istats_.key_domain_min_);
    std::swap(istats_.key_domain_max_, other.istats_.key_domain_max_);
    std::swap(superroot_, other.superroot_);
    std::swap(root_node_, other.root_node_);
  }

 private:
  // Deep copy of tree starting at given node
  // ALEX SHOULDN'T BE WORKED BY OTHER THREADS IN THIS CASE.
  node_type* copy_tree_recursive(const node_type* node) {
    if (!node) return nullptr;
    if (node->is_leaf_) {
      return new (data_node_allocator().allocate(1))
          data_node_type(*static_cast<const data_node_type*>(node));
    } else {
      auto node_copy = new (model_node_allocator().allocate(1))
          model_node_type(*static_cast<const model_node_type*>(node));
      int cur = 0;
      while (cur < node_copy->num_children_) {
        node_type* child_node = node_copy->children_[cur];
        node_type* child_node_copy = copy_tree_recursive(child_node);
        int repeats = 1 << child_node_copy->duplication_factor_;
        for (int i = cur; i < cur + repeats; i++) {
          node_copy->children_[i] = child_node_copy;
        }
        cur += repeats;
      }
      return node_copy;
    }
  }

 public:
  // When bulk loading, Alex can use provided knowledge of the expected fraction
  // of operations that will be inserts
  // For simplicity, operations are either point lookups ("reads") or inserts
  // ("writes)
  // i.e., 0 means we expect a read-only workload, 1 means write-only
  // This is only useful if you set it before bulk loading
  void set_expected_insert_frac(double expected_insert_frac) {
    assert(expected_insert_frac >= 0 && expected_insert_frac <= 1);
    params_.expected_insert_frac = expected_insert_frac;
  }

  // Maximum node size, in bytes.
  // Higher values result in better average throughput, but worse tail/max
  // insert latency.
  void set_max_node_size(int max_node_size) {
    assert(max_node_size >= sizeof(V));
    params_.max_node_size = max_node_size;
    derived_params_.max_fanout = params_.max_node_size / sizeof(void*);
    derived_params_.max_data_node_slots = params_.max_node_size / sizeof(V);
  }

  // Bulk load faster by using sampling to train models.
  // This is only useful if you set it before bulk loading.
  void set_approximate_model_computation(bool approximate_model_computation) {
    params_.approximate_model_computation = approximate_model_computation;
  }

  // Bulk load faster by using sampling to compute cost.
  // This is only useful if you set it before bulk loading.
  void set_approximate_cost_computation(bool approximate_cost_computation) {
    params_.approximate_cost_computation = approximate_cost_computation;
  }

  /*** General helpers ***/

 public:
// Return the data node that contains the key (if it exists).
// Also optionally return the traversal path to the data node.
// traversal_path should be empty when calling this function.
// The returned traversal path begins with superroot and ends with the data
// node's parent.
// Mode 0 : It's for looking the existing key. It should check boundaries.
// Mode 1 : It's for inserting new key. It checks boundaries, but could extend it.
#if ALEX_SAFE_LOOKUP
  forceinline data_node_type* get_leaf(
      AlexKey<T> key, const uint32_t worker_id,
      int mode = 1, std::vector<TraversalNode<T, P>>* traversal_path = nullptr) {
    if (traversal_path) {
      traversal_path->push_back({superroot_, 0});
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "traveling from root" << std::endl;
    alex::coutLock.unlock();
#endif
    node_type* cur = root_node_;
    if (cur->is_leaf_) {
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "root is data node" << std::endl;
      alex::coutLock.unlock();
#endif
      return static_cast<data_node_type*>(cur);
    }

    while (true) {
      auto node = static_cast<model_node_type*>(cur);
Initialization:
      node_type **cur_children = node->children_.read();
      node_type **prev_children = nullptr;
      int num_children = node->num_children_;
      double bucketID_prediction = node->model_.predict_double(key);
      int bucketID = static_cast<int>(bucketID_prediction);
      int dir = 0; //direction of seraching between buckets. 1 for right, -1 for left.
      bucketID =
          std::min<int>(std::max<int>(bucketID, 0), num_children - 1);
      cur = cur_children[bucketID];
      memory_fence();

#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "current bucket : " << bucketID << std::endl;
        //std::cout << "current key max length : " << key.max_key_length_ << std::endl;
        //std::cout << "lb max length : " << cur->min_key_.val_->max_key_length_ << std::endl;
        //std::cout << "ub max length : " << cur->max_key_.val_->max_key_length_ << std::endl;
        std::cout << "min_key : " << cur->min_key_.val_->key_arr_ << std::endl;
        std::cout << "max_key : " << cur->max_key_.val_->key_arr_ << std::endl;
        alex::coutLock.unlock();
#endif

      AlexKey<T> min_tmp_key(istats_.key_domain_min_, max_key_length_);
      AlexKey<T> max_tmp_key(istats_.key_domain_max_, max_key_length_);
      AlexKey<T> *cur_node_min_key = cur->min_key_.read();
      memory_fence();
      AlexKey<T> *cur_node_max_key = cur->max_key_.read();
      memory_fence();
      int was_walking_in_empty = 0;
      int smaller_than_min = key_less_(key, *(cur_node_min_key));
      int larger_than_max = key_less_(*(cur_node_max_key), key);

      if (mode == 0) {//for lookup related get_leaf
        while (smaller_than_min || larger_than_max) {
          if (smaller_than_min && larger_than_max) {
            //empty node. move according to direction.
            //could start at empty node, in this case, move left (since larger key is not possible)
            //SHOULD FIND OUT FAST SEARCHING USING NUMBER OF DUPLICATE POINTER
            was_walking_in_empty = 1;
            if (dir == -1) {
              if (bucketID == 0) {return nullptr;} //out of bound
              bucketID -= 1;
              dir = -1;
            }
            else {
              if (bucketID == num_children-1) {return nullptr;} //out of bound
              bucketID += 1;
              dir = 1;
            }
          }
          else if (smaller_than_min) {
            if (bucketID == 0) {return nullptr;}
            if (dir == 1) {
              //it could be the case where it started from empty node, and initialized direction was wrong
              //in this case, we allow to go backward.
              if (!was_walking_in_empty) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "yo infinite loop baby!" << std::endl;
                alex::coutLock.unlock();
#endif
                return nullptr;
              }
            }
            bucketID -= 1;
            dir = -1;
            was_walking_in_empty = 0;
          }
          else if (larger_than_max) {
            if (bucketID == num_children-1) {return nullptr;}
            if (dir == -1) {
              if (!was_walking_in_empty) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "yo infinite loop baby!" << std::endl;
                alex::coutLock.unlock();
#endif
              }
            }
            bucketID += 1;
            dir = 1;
            was_walking_in_empty = 0;
          }

#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "bucket moved to " << bucketID << std::endl;
          alex::coutLock.unlock();
#endif
          cur = cur_children[bucketID];
          memory_fence();
          cur_node_min_key = cur->min_key_.read();
          memory_fence();
          cur_node_max_key = cur->max_key_.read();
          memory_fence();
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "bucket's min_key is " << cur_node_min_key->key_arr_
                    << " and max_key is " << cur_node_max_key->key_arr_ << std::endl;
          alex::coutLock.unlock();
#endif
          smaller_than_min = key_less_(key, *(cur_node_min_key));
          larger_than_max = key_less_(*(cur_node_max_key), key);
        }
      }
      else if (mode == 1) { //for insert.
        /*we need to check if inserting the key won't make collision with other node's boundary.
          If it does, we need to move to another bucket and insert it. */
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "validating insertion" << std::endl;
        alex::coutLock.unlock();
#endif
        //we first go all the way to left until min_key is smaller than our key.
        while (smaller_than_min) {
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "we are smaller than min (bucket ID : " << bucketID << ")" << std::endl;
          alex::coutLock.unlock();
#endif
          if (bucketID == 0) {return nullptr;}
          bucketID -= 1;
          rcu_progress(worker_id);
          prev_children = cur_children;
          cur_children = node->children_.read();
          if (prev_children != cur_children) {
            //metadata changed -> model node structure changed
            //restart search according to new model
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "metadata changed, restarting insertion." << std::endl;
            alex::coutLock.unlock();
#endif
            goto Initialization;
            rcu_progress(worker_id);
          }
          else { //continue with same metadata.
            cur = cur_children[bucketID]; 
            memory_fence();
            cur_node_min_key = cur->min_key_.read();
            memory_fence();
            cur_node_max_key = cur->max_key_.read();
            memory_fence();
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "continuing search where min/max key is " 
                      << cur_node_min_key->key_arr_ << " " << cur_node_max_key->key_arr_ << std::endl;
            alex::coutLock.unlock();
#endif
            smaller_than_min = key_less_(key, *(cur_node_min_key));
            larger_than_max = key_less_(*(cur_node_max_key), key);
          }
        }
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "found node whose min key is smaller than current key (bucket ID : " << bucketID << ")" << std::endl;
        alex::coutLock.unlock();
#endif
        cur->min_key_.lock();
        memory_fence();
        cur->max_key_.lock();
        memory_fence();
        cur_node_min_key = cur->min_key_.val_;
        memory_fence();
        cur_node_max_key = cur->max_key_.val_;
        memory_fence();
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "node's min_key is " << cur_node_min_key->key_arr_
                  << " and max key is " << cur_node_max_key->key_arr_ << std::endl;
        alex::coutLock.unlock();
#endif
        smaller_than_min = key_less_(key, *(cur_node_min_key));
        larger_than_max = key_less_(*(cur_node_max_key), key);

        if (larger_than_max) { 
          //from here, we won't be reading new metadata values, so we don't rcu_progress.
          //we may read new lower level nodes or their boundaries,
          //but theoretically it won't effect the semantic
          while (true) {
            //we go on finding the next node (that should have larger keys)
            node_type *cur_next;
            int cur_bucketID = bucketID;
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "we are larger than max (bucket ID : " << bucketID << ")" << std::endl;
            alex::coutLock.unlock();
#endif
            do {
              bucketID++;
              if (bucketID > num_children - 1) {
                //std::cout << "for debugging touchdown!" << std::endl;
                break;
              }
              cur_next = cur_children[bucketID];
              if (cur_next != cur) {break;}
            } while (true);

            if (bucketID > num_children - 1) {
              //should EXTEND the last node.
              AlexKey<T> *new_max_key = new AlexKey<T>(max_key_length_);
              std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
              AlexKey<T> *old_max_key = cur->max_key_.val_;
              cur->max_key_.val_ = new_max_key;
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              rcu_barrier(worker_id);
              delete old_max_key;
              //when calling rcu_barrier, structure of model node could have changed
              //or our original leaf node could have changed to new leaf node.
              //we should check it, and if it changed... we should do the whole search again(...)
              prev_children = cur_children;
              cur_children = node->children_.read();
              if (prev_children != cur_children) {
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "children changed" << std::endl;
              alex::coutLock.unlock();
#endif
                rcu_progress(worker_id);
                goto Initialization;
              }
              node_type *after_rcu_cur_next = cur_children[cur_bucketID];
              if (after_rcu_cur_next != cur) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "metadata changed" << std::endl;
                alex::coutLock.unlock();
#endif
                rcu_progress(worker_id);
                goto Initialization;
              }
              bucketID = cur_bucketID;
              break;
            }

            //If we found the new node, we try to obtain the lock of those nodes.
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "found new node (bucket ID : " << bucketID << ")" << std::endl;
            alex::coutLock.unlock();
#endif
            AlexKey<T> *next_node_min_key, *next_node_max_key;
            cur_next->min_key_.lock();
            memory_fence();
            cur_next->max_key_.lock();
            memory_fence();
            next_node_min_key = cur_next->min_key_.val_;
            memory_fence();
            next_node_max_key = cur_next->max_key_.val_;
            memory_fence();
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "node's min_key is " << next_node_min_key->key_arr_
                      << " and max key is " << next_node_max_key->key_arr_ << std::endl;
            alex::coutLock.unlock();
#endif
            smaller_than_min = key_less_(key, *(next_node_min_key));
            larger_than_max = key_less_(*(next_node_max_key), key);
              
            if (smaller_than_min && larger_than_max) {
              // next node was empty node
              // we again need to search the node after this empty node.
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "new node was empty node (bucket ID : " << bucketID << ")" << std::endl;
              alex::coutLock.unlock();
#endif
EmptyNodeStart:
              node_type *cur_dbl_next;
              int cur_next_bucketID = bucketID;
              do {
                bucketID++;
                if (bucketID > num_children - 1) {
                  break; //special case.
                }
                cur_dbl_next = cur_children[bucketID];
                if (cur_next != cur_dbl_next) {break;}
              } while (true);

              if (bucketID > num_children - 1) {
                //need to insert in empty data node...
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "empty node was last node (shouldn't happen in normal situation)" << std::endl;
                alex::coutLock.unlock();
#endif
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur = cur_next;
                bucketID = cur_next_bucketID;
                break;
              }

              //If we found the new node, and if we're not in special case,
              //we try to obtain the lock of those nodes.
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "we found another node (next next node) (bucket ID : " << bucketID << ")" << std::endl;
              alex::coutLock.unlock();
#endif
              AlexKey<T> *dbl_next_node_min_key, *dbl_next_node_max_key;
              cur_dbl_next->min_key_.lock();
              memory_fence();
              cur_dbl_next->max_key_.lock();
              memory_fence();
              dbl_next_node_min_key = cur_dbl_next->min_key_.val_;
              memory_fence();
              dbl_next_node_max_key = cur_dbl_next->max_key_.val_;
              memory_fence();
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "node's min_key is " << dbl_next_node_min_key->key_arr_
                        << " and max key is " << dbl_next_node_max_key->key_arr_ << std::endl;
              alex::coutLock.unlock();
#endif
              smaller_than_min = key_less_(key, *(dbl_next_node_min_key));
              larger_than_max = key_less_(*(dbl_next_node_max_key), key);

              if (smaller_than_min && larger_than_max) {
                //it's another empty data node
                //decided to have that new empty data node as cur_next.
                //there is no special reason of choosing that node as cur_next
                //but searching must continue becuase of possible wrong boundary.
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "another empty data node found. " << std::endl;
                alex::coutLock.unlock();
#endif
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_next = cur_dbl_next;
                goto EmptyNodeStart;
              }
              else if (smaller_than_min) {
                //try inserting to empty data node. which means, we enter the empty data node!
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "decided to insert to empty data node (it's bucket ID : " << cur_next_bucketID << ")" << std::endl;
                alex::coutLock.unlock();
#endif
                AlexKey<T> *new_max_key = new AlexKey<T>(max_key_length_);
                AlexKey<T> *new_min_key = new AlexKey<T>(max_key_length_);
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
                std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_min_key->key_arr_);
                AlexKey<T> *old_max_key = cur_next->max_key_.val_;
                AlexKey<T> *old_min_key = cur_next->min_key_.val_;
                cur_next->min_key_.val_ = new_min_key;
                cur_next->max_key_.val_ = new_max_key;
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                rcu_barrier(worker_id);
                delete old_max_key;
                delete old_min_key;
                //when calling rcu_barrier, structure of model node could have changed
                //or our original leaf node could have changed to new leaf node.
                //we should check it, and if it changed... we should do the whole search again(...)
                prev_children = cur_children;
                cur_children = node->children_.read();
                if (prev_children != cur_children) {
#if DEBUG_PRINT
                  alex::coutLock.lock();
                  std::cout << "t" << worker_id << " - ";
                  std::cout << "children changed" << std::endl;
                  alex::coutLock.unlock();
#endif
                  rcu_progress(worker_id);
                  goto Initialization;
                }
                node_type *after_rcu_cur_next = cur_children[cur_next_bucketID];
                if (after_rcu_cur_next != cur_next) {
#if DEBUG_PRINT
                  alex::coutLock.lock();
                  std::cout << "t" << worker_id << " - ";
                  std::cout << "metadata changed" << std::endl;
                  alex::coutLock.unlock();
#endif
                  rcu_progress(worker_id);
                  goto Initialization;
                }
                cur = after_rcu_cur_next;
                bucketID = cur_next_bucketID;
                break;
              }
              if (larger_than_max) {
                //we are even larger than largest key of next node.
                //do the same progress as before, except that cur_dbl_next node is cur node.
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "we are larger than next next node (bucket ID : " << bucketID << ")" << std::endl;
                alex::coutLock.unlock();
#endif
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                cur = cur_dbl_next;
              }
              else {
                //the boundary may have changed to just include our key while moving
                //so choose 'cur_dbl_next' node as our moving node
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "decided to enter next next node (bucket ID : " << bucketID << ")" << std::endl;
                alex::coutLock.unlock();
#endif
                cur->min_key_.unlock();
                cur->max_key_.unlock();
                cur_next->min_key_.unlock();
                cur_next->max_key_.unlock();
                cur_dbl_next->min_key_.unlock();
                cur_dbl_next->max_key_.unlock();
                cur = cur_dbl_next;
                break;
              }
            }
            else if (smaller_than_min) {
              //Doesn't matter to enter 'cur' model node. we also extend it.
              //should do rcu barrier, since some other node may be using that boundary.
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "decided to extend current node and enter (it's bucket ID : " << cur_bucketID << ")" << std::endl;
              alex::coutLock.unlock();
#endif
              AlexKey<T> *new_max_key = new AlexKey<T>(max_key_length_);
              std::copy(key.key_arr_, key.key_arr_ + max_key_length_, new_max_key->key_arr_);
              AlexKey<T> *old_max_key = cur->max_key_.val_;
              cur->max_key_.val_ = new_max_key;
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur_next->min_key_.unlock();
              cur_next->max_key_.unlock();
              rcu_barrier(worker_id);
              delete old_max_key;
              //when calling rcu_barrier, structure of model node could have changed
              //or our original leaf node could have changed to new leaf node.
              //we should check it, and if it changed... we should do the whole search again(...)
              prev_children = cur_children;
              cur_children = node->children_.read();
              if (prev_children != cur_children) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "metadata changed" << std::endl;
                alex::coutLock.unlock();
#endif
                rcu_progress(worker_id);
                goto Initialization;
              }
              node_type *after_rcu_cur = cur_children[cur_bucketID];
              if (after_rcu_cur != cur) {
#if DEBUG_PRINT
                alex::coutLock.lock();
                std::cout << "t" << worker_id << " - ";
                std::cout << "node changed" << std::endl;
                alex::coutLock.unlock();
#endif
                rcu_progress(worker_id);
                goto Initialization;
              }
              cur = after_rcu_cur;
              bucketID = cur_bucketID;
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "update finished" << std::endl;
              alex::coutLock.unlock();
#endif
              break;
            }
            else if (larger_than_max) {
              //we are even larger than largest key of next node.
              //do the same progress as before, except that cur_next node is cur node.
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "we are larger than next node (bucket ID : " << bucketID << ")" << std::endl;
              alex::coutLock.unlock();
#endif
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur = cur_next;
            }
            else {
              //the boundary may have changed to just include our key while moving
              //so choose 'cur_next' node as our moving node
#if DEBUG_PRINT
              alex::coutLock.lock();
              std::cout << "t" << worker_id << " - ";
              std::cout << "decided to enter next node (it's bucket ID : " << bucketID << ")" << std::endl;
              alex::coutLock.unlock();
#endif
              cur->min_key_.unlock();
              cur->max_key_.unlock();
              cur_next->min_key_.unlock();
              cur_next->max_key_.unlock();
              cur = cur_next;
              break;
            }
          }
        }
        else { //this is the next node we'll enter.
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "decided to enter current node (it's bucket ID : " << bucketID << ")" << std::endl;
          alex::coutLock.unlock();
#endif
          cur->min_key_.unlock();
          cur->max_key_.unlock();
        }
      }
      if (traversal_path) {
        traversal_path->push_back({node, bucketID});
      }
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "this leaf has min_key_ as " << cur->min_key_.val_->key_arr_ 
                << " and max_key_ as " << cur->max_key_.val_->key_arr_ << std::endl;
      alex::coutLock.unlock();
#endif
      if (cur->is_leaf_) {
        stats_.num_node_lookups.add(cur->level_);
        // we don't do rcu_progress here, since we are entering data node.
        // rcu_progress should be called at adequate point where the users finished using this data node.
        // If done ignorantly, it could cause null pointer access (because of destruction by other thread)
        return (data_node_type *) cur;
      }
      //entering model node, need to progress
      //chosen model nodes are never destroyed, (without erase implementation, not used currently.)
      //Synchronization issue will be checked by another while loop.
      rcu_progress(worker_id);
    }
  }
#else
  data_node_type* get_leaf(
      AlexKey<T> key, std::vector<TraversalNode>* traversal_path = nullptr) const {
    return nullptr; //not implemented
  }
#endif

 private:
  // Honestly, can't understand why below 4 functions exists 
  // (first_data_node / last_data_node / get_min_key / get_max_key)
  // (since it's declared private and not used anywhere)
  // Return left-most data node
  data_node_type* first_data_node() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_.val_[0];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Return right-most data node
  data_node_type* last_data_node() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto node = static_cast<model_node_type*>(cur);
      cur = node->children_.val_[node->num_children_ - 1];
    }
    return static_cast<data_node_type*>(cur);
  }

  // Returns minimum key in the index
  T *get_min_key() const { return first_data_node()->first_key(); }

  // Returns maximum key in the index
  T *get_max_key() const { return last_data_node()->last_key(); }

  // Link all data nodes together. Used after bulk loading.
  void link_all_data_nodes() {
    data_node_type* prev_leaf = nullptr;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      node_type* cur = node_it.current();
      if (cur->is_leaf_) {
        auto node = static_cast<data_node_type*>(cur);
        if (prev_leaf != nullptr) {
          prev_leaf->next_leaf_.val_ = node;
          node->prev_leaf_.val_ = prev_leaf;
        }
        prev_leaf = node;
      }
    }
  }

  // Link the new data nodes together when old data node is replaced by two new
  // data nodes.
  void link_data_nodes(data_node_type* old_leaf,
                       data_node_type* left_leaf, data_node_type* right_leaf) {
    data_node_type *old_leaf_prev_leaf = old_leaf->prev_leaf_.read();
    data_node_type *old_leaf_next_leaf = old_leaf->next_leaf_.read();
    if (old_leaf_prev_leaf != nullptr) {
      data_node_type *olpl_pending_rl = old_leaf_prev_leaf->pending_right_leaf_.read();
      if (olpl_pending_rl != nullptr) {
        olpl_pending_rl->next_leaf_.update(left_leaf);
        left_leaf->prev_leaf_.update(olpl_pending_rl);
      }
      else {
        old_leaf_prev_leaf->next_leaf_.update(left_leaf);
        left_leaf->prev_leaf_.update(old_leaf_prev_leaf);
      }
    }
    else {
      left_leaf->prev_leaf_.update(nullptr);
    }
    left_leaf->next_leaf_.update(right_leaf);
    right_leaf->prev_leaf_.update(left_leaf);
    if (old_leaf_next_leaf != nullptr) {
      data_node_type *olnl_pending_ll = old_leaf_next_leaf->pending_left_leaf_.read();
      if (olnl_pending_ll != nullptr) {
        olnl_pending_ll->prev_leaf_.update(right_leaf);
        right_leaf->next_leaf_.update(olnl_pending_ll);
      }
      else {
        old_leaf_next_leaf->prev_leaf_.update(right_leaf);
        right_leaf->next_leaf_.update(old_leaf_next_leaf);
      }
    }
    else {
      right_leaf->next_leaf_.update(nullptr);
    }
  }

  /*** Allocators and comparators ***/

 public:
  Alloc get_allocator() const { return allocator_; }

  Compare key_comp() const { return key_less_; }

 private:
  typename model_node_type::alloc_type model_node_allocator() {
    return typename model_node_type::alloc_type(allocator_);
  }

  typename data_node_type::alloc_type data_node_allocator() {
    return typename data_node_type::alloc_type(allocator_);
  }

  typename model_node_type::pointer_alloc_type pointer_allocator() {
    return typename model_node_type::pointer_alloc_type(allocator_);
  }

  void delete_node(node_type* node) {
    if (node == nullptr) {
      return;
    } else if (node->is_leaf_) {
      data_node_allocator().destroy(static_cast<data_node_type*>(node));
      data_node_allocator().deallocate(static_cast<data_node_type*>(node), 1);
    } else {
      model_node_allocator().destroy(static_cast<model_node_type*>(node));
      model_node_allocator().deallocate(static_cast<model_node_type*>(node), 1);
    }
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Bulk loading ***/

 public:
  // values should be the sorted array of key-payload pairs.
  // The number of elements should be num_keys.
  // The index must be empty when calling this method.
  void bulk_load(const V values[], int num_keys) {
    if (stats_.num_keys.val_ > 0 || num_keys <= 0) {
      return;
    }
    delete_node(root_node_);  // delete the empty root node from constructor

    stats_.num_keys.val_ = num_keys;

    // Build temporary root model, which outputs a CDF in the range [0, 1]
    root_node_ =
        new (model_node_allocator().allocate(1)) model_node_type(0, nullptr, max_key_length_, allocator_);
    AlexKey<T> min_key = values[0].first;
    AlexKey<T> max_key = values[num_keys - 1].first;

    if (typeid(T) == typeid(char)) { //for string key
      LinearModelBuilder<T> root_model_builder(&(root_node_->model_));
      for (int i = 0; i < num_keys; i++) {
#if DEBUG_PRINT
        printf("adding : %f\n", (double) (i) / (num_keys-1));
#endif
        root_model_builder.add(values[i].first, (double) (i) / (num_keys-1));
      }
      root_model_builder.build();
    }
    else { //for numeric key
      std::cout << "Please use only string keys" << std::endl;
      abort();
    }
#if DEBUG_PRINT
    for (int i = 0; i < num_keys; i++) {
      std::cout << values[i].first.key_arr_ << " prediction "
                << root_node_->model_.predict_double(values[i].first) 
                << std::endl;
    }
    std::cout << "left prediction result (bulk_load) " 
              << root_node_->model_.predict_double(values[1].first) 
              << std::endl;
    std::cout << "right prediction result (bulk_load) " 
              << root_node_->model_.predict_double(values[num_keys-2].first) 
              << std::endl;
#endif

    // Compute cost of root node
    LinearModel<T> root_data_node_model(max_key_length_);
    data_node_type::build_model(values, num_keys, &root_data_node_model,
                                params_.approximate_model_computation);
    DataNodeStats stats;
    root_node_->cost_ = data_node_type::compute_expected_cost(
        values, num_keys, data_node_type::kInitDensity_,
        params_.expected_insert_frac, &root_data_node_model,
        params_.approximate_cost_computation, &stats);

    // Recursively bulk load
    bulk_load_node(values, num_keys, root_node_, nullptr, num_keys,
                   &root_data_node_model);

    if (root_node_->is_leaf_) {
      static_cast<data_node_type*>(root_node_)
          ->expected_avg_exp_search_iterations_ = stats.num_search_iterations;
      static_cast<data_node_type*>(root_node_)->expected_avg_shifts_ =
          stats.num_shifts;
    }

    create_superroot();
    update_superroot_key_domain();
    link_all_data_nodes();

#if DEBUG_PRINT
    std::cout << "structure's min_key after bln : " << istats_.key_domain_min_ << std::endl;
    std::cout << "structure's max_key after bln : " << istats_.key_domain_max_ << std::endl;
#endif
  }

 private:
  // Only call this after creating a root node
  void create_superroot() {
    if (!root_node_) return;
    delete_node(superroot_);
    superroot_ = new (model_node_allocator().allocate(1))
        model_node_type(static_cast<short>(root_node_->level_ - 1), nullptr, max_key_length_, allocator_);
    superroot_->num_children_ = 1;
    superroot_->children_.val_ =
        new (pointer_allocator().allocate(1)) node_type*[1];
    update_superroot_pointer();
  }

  // Updates the key domain based on the min/max keys and retrains the model.
  // Should only be called immediately after bulk loading
  void update_superroot_key_domain() {
    assert(stats_.num_inserts.val_ == 0 || root_node_->is_leaf_);
    T *min_key_arr, *max_key_arr;
    //min/max should always be '!' and '~...~'
    //the reason we are doing this cumbersome process is because
    //'!' may not be inserted at the first data node.
    //We need some way to handle this. May be fixed by unbiasing keys.
    min_key_arr = (T *) malloc(max_key_length_);
    max_key_arr = (T *) malloc(max_key_length_);
    for (unsigned int i = 0; i < max_key_length_; i++) {
      max_key_arr[i] = STR_VAL_MAX;
      min_key_arr[i] = (i == 0) ? STR_VAL_MIN : 0;
    }

#if DEBUG_PRINT
    for (unsigned int i = 0; i < max_key_length_; i++) {
      std::cout << min_key_arr[i] << ' ';
    }
    std::cout << std::endl;
    for (unsigned int i = 0; i < max_key_length_; i++) {
      std::cout << max_key_arr[i] << ' ';
    }
    std::cout << std::endl;
#endif
    std::copy(min_key_arr, min_key_arr + max_key_length_, istats_.key_domain_min_);
    std::copy(max_key_arr, max_key_arr + max_key_length_, istats_.key_domain_max_);

    AlexKey<T> mintmpkey(istats_.key_domain_min_, max_key_length_);
    AlexKey<T> maxtmpkey(istats_.key_domain_max_, max_key_length_);
    if (key_equal(mintmpkey, maxtmpkey)) {//keys are equal
      unsigned int non_zero_cnt_ = 0;

      for (unsigned int i = 0; i < max_key_length_; i++) {
        if (istats_.key_domain_min_[i] == 0) {
          superroot_->model_.a_[i] = 0;
        }
        else {
          superroot_->model_.a_[i] = 1 / istats_.key_domain_min_[i];
          non_zero_cnt_ += 1;
        }
      }
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        superroot_->model_.a_[i] /= non_zero_cnt_;
      }
      superroot_->model_.b_ = 0;
    }
    else {//keys are not equal
      double direction_vector_[max_key_length_] = {0.0};
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        direction_vector_[i] = (double) istats_.key_domain_max_[i] - istats_.key_domain_min_[i];
      }
      superroot_->model_.b_ = 0.0;
      unsigned int non_zero_cnt_ = 0;
      for (unsigned int i = 0; i < max_key_length_; i++) {
        if (direction_vector_[i] == 0) {
          superroot_->model_.a_[i] = 0;
        }
        else {
          superroot_->model_.a_[i] = 1 / (direction_vector_[i]);
          superroot_->model_.b_ -= istats_.key_domain_min_[i] / direction_vector_[i];
          non_zero_cnt_ += 1;
        }
      }
      
      for (unsigned int i = 0; i < max_key_length_; i++) {
        superroot_->model_.a_[i] /= non_zero_cnt_;
      }
      superroot_->model_.b_ /= non_zero_cnt_;
    }

    if (typeid(T) == typeid(char)) { //need to free malloced objects.
      free(min_key_arr);
      free(max_key_arr);
    }

#if DEBUG_PRINT
    std::cout << "left prediction result (uskd) " << superroot_->model_.predict_double(mintmpkey) << std::endl;
    std::cout << "right prediction result (uskd) " << superroot_->model_.predict_double(maxtmpkey) << std::endl;
#endif
  }

  void update_superroot_pointer() {
    superroot_->children_.val_[0] = root_node_;
    superroot_->level_ = static_cast<short>(root_node_->level_ - 1);
  }

  // Recursively bulk load a single node.
  // Assumes node has already been trained to output [0, 1), has cost.
  // Figures out the optimal partitioning of children.
  // node is trained as if it's a model node.
  // data_node_model is what the node's model would be if it were a data node of
  // dense keys.
  void bulk_load_node(const V values[], int num_keys, node_type*& node,
                      model_node_type* parent, int total_keys,
                      const LinearModel<T>* data_node_model = nullptr) {
    // Automatically convert to data node when it is impossible to be better
    // than current cost
#if DEBUG_PRINT
    std::cout << "called bulk_load_node!" << std::endl;
#endif
    if (num_keys <= derived_params_.max_data_node_slots *
                        data_node_type::kInitDensity_ &&
        (node->cost_ < kNodeLookupsWeight || node->model_.a_ == 0)) {
      stats_.num_data_nodes.increment();
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         node->max_key_length_, parent,
                         key_less_, allocator_);
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation);
      data_node->cost_ = node->cost_;
      delete_node(node);
      node = data_node;
#if DEBUG_PRINT
      std::cout << "returned because it can't be better" << std::endl;
#endif
      return;
    }

    // Use a fanout tree to determine the best way to divide the key space into
    // child nodes
    std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;
    std::pair<int, double> best_fanout_stats;
    int max_data_node_keys = static_cast<int>(
        derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
    best_fanout_stats = fanout_tree::find_best_fanout_bottom_up<T, P>(
        values, num_keys, node, total_keys, used_fanout_tree_nodes,
        derived_params_.max_fanout, max_data_node_keys,
        params_.expected_insert_frac, params_.approximate_model_computation,
        params_.approximate_cost_computation);
    int best_fanout_tree_depth = best_fanout_stats.first;
    double best_fanout_tree_cost = best_fanout_stats.second;

    // Decide whether this node should be a model node or data node
    if (best_fanout_tree_cost < node->cost_ ||
        num_keys > derived_params_.max_data_node_slots *
                       data_node_type::kInitDensity_) {
#if DEBUG_PRINT
      std::cout << "decided that current bulk_load_node calling node should be model node" << std::endl;
#endif
      // Convert to model node based on the output of the fanout tree
      stats_.num_model_nodes.increment();
      auto model_node = new (model_node_allocator().allocate(1))
          model_node_type(node->level_, parent, max_key_length_, allocator_);
      if (best_fanout_tree_depth == 0) {
        // slightly hacky: we assume this means that the node is relatively
        // uniform but we need to split in
        // order to satisfy the max node size, so we compute the fanout that
        // would satisfy that condition
        // in expectation
        best_fanout_tree_depth =
            static_cast<int>(std::log2(static_cast<double>(num_keys) /
                                       derived_params_.max_data_node_slots)) +
            1;
        //clear pointers used in fanout_tree (O(N)), and then empty used_fanout_tree_nodes.
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
          delete[] tree_node.a;
        }
        used_fanout_tree_nodes.clear();
        int max_data_node_keys = static_cast<int>(
            derived_params_.max_data_node_slots * data_node_type::kInitDensity_);
#if DEBUG_PRINT
        std::cout << "computing level for depth" << std::endl;
#endif
        fanout_tree::compute_level<T, P>(
            values, num_keys, node, total_keys, used_fanout_tree_nodes,
            best_fanout_tree_depth, max_data_node_keys,
            params_.expected_insert_frac, params_.approximate_model_computation,
            params_.approximate_cost_computation);
#if DEBUG_PRINT
        std::cout << "finished level computing" << std::endl;
#endif
      }
      int fanout = 1 << best_fanout_tree_depth;
#if DEBUG_PRINT
      std::cout << "chosen fanout is... : " << fanout << std::endl;
#endif
      //obtianing CDF resulting to [0,fanout]
      LinearModel<T> tmp_model(node->max_key_length_);
      LinearModelBuilder<T> tmp_model_builder(&tmp_model);
      for (int i = 0; i < num_keys; i++) {
        tmp_model_builder.add(values[i].first, ((double) i * fanout / (num_keys-1)));
      }
      tmp_model_builder.build();
      for (unsigned int i = 0; i < node->model_.max_key_length_; i++) {
        model_node->model_.a_[i] = tmp_model.a_[i];
      }
      model_node->model_.b_ = tmp_model.b_; 
      

      model_node->num_children_ = fanout;
      model_node->children_.val_ =
          new (pointer_allocator().allocate(fanout)) node_type*[fanout];

      // Instantiate all the child nodes and recurse
      int cur = 0;
      int idx = 0, f_idx = 0, l_idx = 0;
#if DEBUG_PRINT
      int cumu_repeat = 0;
#endif
      for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
        auto child_node = new (model_node_allocator().allocate(1))
            model_node_type(static_cast<short>(node->level_ + 1), model_node, max_key_length_, allocator_);
        child_node->cost_ = tree_node.cost;
        child_node->duplication_factor_ =
            static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
        int repeats = 1 << child_node->duplication_factor_;
        double left_value, right_value;
        left_value = cur;
        right_value = cur + repeats;
#if DEBUG_PRINT
        cumu_repeat += repeats;
        std::cout << "started finding boundary..." << std::endl;
        std::cout << "for left_value with : " << left_value << std::endl;
        std::cout << "and right_value with : " << right_value << std::endl;
        std::cout << "so covering indexes are : " << cumu_repeat - repeats << "~" << cumu_repeat - 1 << std::endl;
#endif
        double left_boundary[node->max_key_length_];
        double right_boundary[node->max_key_length_];
        //slightly hacky
        // It tries to find the first value larger or equal to left / right value.
        // Then assumes those are the left/right boundary.
        for (; idx < num_keys; idx++) {
#if DEBUG_PRINT
          std::cout << values[idx].first.key_arr_ << " predicted as " << model_node->model_.predict_double(values[idx].first) << std::endl;
#endif
          if (model_node->model_.predict_double(values[idx].first) >= left_value) {
            for (unsigned int i = 0; i < model_node->max_key_length_; i++) {
              left_boundary[i] = (double) values[idx].first.key_arr_[i];
            }
            f_idx = idx;
            break;
          }
        }
        if (idx == num_keys) {
          for (unsigned int i = 0; i < model_node->max_key_length_; i++) {
            left_boundary[i] = (double) values[num_keys-1].first.key_arr_[i];
          }
          f_idx = num_keys - 1;
        }
        for (; idx < num_keys; idx++) {
#if DEBUG_PRINT
          std::cout << values[idx].first.key_arr_ << " predicted as " << model_node->model_.predict_double(values[idx].first) << std::endl;
#endif
          if (model_node->model_.predict_double(values[idx].first) >= right_value) {
            for (unsigned int i = 0; i < model_node->max_key_length_; i++) {
              right_boundary[i] = (double) values[idx].first.key_arr_[i];
            }
            l_idx = idx;
            break;
          }
        }
        if (idx == num_keys) {
          for (unsigned int i = 0; i < model_node->max_key_length_; i++) {
            right_boundary[i] = (double) values[num_keys-1].first.key_arr_[i];
          }
          l_idx = num_keys - 1;
        }
#if DEBUG_PRINT
        std::cout << "finished finding boundary..." << std::endl;
        if (typeid(T) != typeid(char)) {
          std::cout << "left boundary is : " << std::setprecision (17) << left_boundary[0] << std::endl;
          std::cout << "right boundary is : " << std::setprecision (17) << right_boundary[0] << std::endl;
        }
        else {
          std::cout << "left boundary is : ";
          for (unsigned int i = 0; i < node->max_key_length_; i++) {std::cout << (char) left_boundary[i];}
          std::cout << std::endl;
          std::cout << "right boundary is : ";
          for (unsigned int i = 0; i < node->max_key_length_; i++) {std::cout << (char) right_boundary[i];}
          std::cout << std::endl;
        }
#endif

        //obtain CDF with range [0,1]
        if (typeid(T) == typeid(char)) {
          int num_keys = l_idx - f_idx + 1;
          LinearModelBuilder<T> child_model_builder(&child_node->model_);
#if DEBUG_PRINT
          printf("l_idx : %d, f_idx : %d, num_keys : %d\n", l_idx, f_idx, num_keys);
#endif
          if (num_keys == 1) {
            child_model_builder.add(values[f_idx].first, 1.0);
          }
          else {
            for (int i = f_idx; i < f_idx + num_keys; i++) {
              child_model_builder.add(values[i].first, (double) (i-f_idx)/(num_keys-1));
            }
          }
          child_model_builder.build();
        }
        else {
          child_node->model_.a_[0] = 1.0 / (right_boundary[0] - left_boundary[0]);
          child_node->model_.b_ = -child_node->model_.a_[0] * left_boundary[0];
        }

#if DEBUG_PRINT
        T left_key[max_key_length_];
        T right_key[max_key_length_];
        for (unsigned int i = 0; i < max_key_length_; i++) {
          left_key[i] = left_boundary[i];
          right_key[i] = right_boundary[i];
        }
        std::cout << "left prediction result (bln) " << child_node->model_.predict_double(AlexKey<T>(left_key, max_key_length_)) << std::endl;
        std::cout << "right prediction result (bln) " << child_node->model_.predict_double(AlexKey<T>(right_key, max_key_length_)) << std::endl;
#endif

        model_node->children_.val_[cur] = child_node;
        LinearModel<T> child_data_node_model(tree_node.a, tree_node.b, max_key_length_);
        bulk_load_node(values + tree_node.left_boundary,
                       tree_node.right_boundary - tree_node.left_boundary,
                       model_node->children_.val_[cur], model_node, total_keys,
                       &child_data_node_model);
        model_node->children_.val_[cur]->duplication_factor_ =
            static_cast<uint8_t>(best_fanout_tree_depth - tree_node.level);
        if (model_node->children_.val_[cur]->is_leaf_) {
          static_cast<data_node_type*>(model_node->children_.val_[cur])
              ->expected_avg_exp_search_iterations_ =
              tree_node.expected_avg_search_iterations;
          static_cast<data_node_type*>(model_node->children_.val_[cur])
              ->expected_avg_shifts_ = tree_node.expected_avg_shifts;
        }
        for (int i = cur + 1; i < cur + repeats; i++) {
          model_node->children_.val_[i] = model_node->children_.val_[cur];
        }
        cur += repeats;
      }

      /* update min_key_, max_key_ for new model node*/
      std::copy(values[0].first.key_arr_, values[0].first.key_arr_ + max_key_length_,
        model_node->min_key_.val_->key_arr_);
      std::copy(values[num_keys-1].first.key_arr_, values[num_keys-1].first.key_arr_ + max_key_length_,
        model_node->max_key_.val_->key_arr_);
      
      
#if DEBUG_PRINT
      std::cout << "min_key_(model_node) : " << model_node->min_key_.val_->key_arr_ << std::endl;
      std::cout << "max_key_(model_node) : " << model_node->max_key_.val_->key_arr_ << std::endl;
      for (int i = 0; i < fanout; i++) {
        std::cout << i << "'s initial pointer value is : " << model_node->children_.val_[i] << std::endl;
        std::cout << i << "'s min_key is : " << model_node->children_.val_[i]->min_key_.val_->key_arr_ << std::endl;
        std::cout << i << "'s max_key is : " << model_node->children_.val_[i]->max_key_.val_->key_arr_ << std::endl;
      }
#endif

      delete_node(node);
      node = model_node;
    } else {
#if DEBUG_PRINT
      std::cout << "decided that current bulk_load_node calling node should be data node" << std::endl;
#endif
      // Convert to data node
      stats_.num_data_nodes.increment();
      auto data_node = new (data_node_allocator().allocate(1))
          data_node_type(node->level_, derived_params_.max_data_node_slots,
                         max_key_length_, parent,
                         key_less_, allocator_);
      data_node->bulk_load(values, num_keys, data_node_model,
                           params_.approximate_model_computation);
      data_node->cost_ = node->cost_;
      delete_node(node);
      node = data_node;
    }

    //empty used_fanout_tree_nodes for preventing memory leakage.
    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
      delete[] tree_node.a;
    }
#if DEBUG_PRINT
    std::cout << "returned using fanout" << std::endl;
#endif
  }

  // Caller needs to set the level, duplication factor, and neighbor pointers of
  // the returned data node
  data_node_type* bulk_load_leaf_node_from_existing(
      const data_node_type* existing_node, int left, int right, uint32_t worker_id,
      bool compute_cost = true, const fanout_tree::FTNode* tree_node = nullptr,
      bool reuse_model = false, bool keep_left = false,
      bool keep_right = false) {
    auto node = new (data_node_allocator().allocate(1))
        data_node_type(max_key_length_, existing_node->parent_, key_less_, allocator_);
    stats_.num_data_nodes.increment();
    if (tree_node) {
      // Use the model and num_keys saved in the tree node so we don't have to
      // recompute it
      LinearModel<T> precomputed_model(tree_node->a, tree_node->b, max_key_length_);
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right, &precomputed_model,
                                    tree_node->num_keys);
    } else if (reuse_model) {
      // Use the model from the existing node
      // Assumes the model is accurate
      int num_actual_keys = existing_node->num_keys_in_range(left, right);
      LinearModel<T> precomputed_model(existing_node->model_);
      precomputed_model.b_ -= left;
      precomputed_model.expand(static_cast<double>(num_actual_keys) /
                               (right - left));
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right, &precomputed_model,
                                    num_actual_keys);
    } else {
      node->bulk_load_from_existing(existing_node, left, right, worker_id, keep_left,
                                    keep_right);
    }
    node->max_slots_ = derived_params_.max_data_node_slots;
    if (compute_cost) {
      node->cost_ = node->compute_expected_cost(existing_node->frac_inserts());
    }
    return node;
  }

  /*** Lookup ***/

 public:
  // Looks for an exact match of the key
  // If the key does not exist, returns an end iterator
  // If there are multiple keys with the same value, returns an iterator to the
  // right-most key
  // If you instead want an iterator to the left-most key with the input value,
  // use lower_bound()
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator find(const AlexKey<T>& key) {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return end();} //error when finding key.
    int idx = leaf->find_key(key);
    if (idx < 0) {
      return end();
    } else {
      return Iterator(leaf, idx);
    }
  }

  typename self_type::ConstIterator find(const AlexKey<T>& key) const {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return cend();} //error when finding key.
    int idx = leaf->find_key(key);
    if (idx < 0) {
      return cend();
    } else {
      return ConstIterator(leaf, idx);
    }
  }

  size_t count(const AlexKey<T>& key) {
    ConstIterator it = lower_bound(key);
    size_t num_equal = 0;
    while (!it.is_end() && key_equal(it.key(), key)) {
      num_equal++;
      ++it;
    }
    return num_equal;
  }

  // Returns an iterator to the first key no less than the input value
  //returns end iterator on error.
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator lower_bound(const AlexKey<T>& key) {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return end();}
    int idx = leaf->find_lower(key);
    return Iterator(leaf, idx);  // automatically handles the case where idx ==
                                 // leaf->data_capacity
  }

  typename self_type::ConstIterator lower_bound(const AlexKey<T>& key) const {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return cend();}
    int idx = leaf->find_lower(key);
    return ConstIterator(leaf, idx);  // automatically handles the case where
                                      // idx == leaf->data_capacity
  }

  // Returns an iterator to the first key greater than the input value
  // returns end iterator on error
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator upper_bound(const AlexKey<T>& key) {
    stats_.num_lookups++;
    data_node_type* leaf = typeid(T) == typeid(char) ? get_leaf(key, 0) : get_leaf(key);
    if (leaf == nullptr) {return end();}
    int idx = leaf->find_upper(key);
    return Iterator(leaf, idx);  // automatically handles the case where idx ==
                                 // leaf->data_capacity
  }

  typename self_type::ConstIterator upper_bound(const AlexKey<T>& key) const {
    stats_.num_lookups++;
    data_node_type* leaf = typeid(T) == typeid(char) ? get_leaf(key, 0) : get_leaf(key);
    if (leaf == nullptr) {return cend();}
    int idx = leaf->find_upper(key);
    return ConstIterator(leaf, idx);  // automatically handles the case where
                                      // idx == leaf->data_capacity
  }

  std::pair<Iterator, Iterator> equal_range(const AlexKey<T>& key) {
    return std::pair<Iterator, Iterator>(lower_bound(key), upper_bound(key));
  }

  std::pair<ConstIterator, ConstIterator> equal_range(const AlexKey<T>& key) const {
    return std::pair<ConstIterator, ConstIterator>(lower_bound(key),
                                                   upper_bound(key));
  }

  // Returns whether payload search was successful, and the payload itself if it was successful.
  // This avoids the overhead of creating an iterator

  std::pair<bool, P> get_payload(const AlexKey<T>& key, int32_t worker_id) {
    stats_.num_lookups.increment();
    data_node_type* leaf = get_leaf(key, worker_id, 0);
    if (leaf == nullptr) {
      rcu_progress(worker_id);
      return {false, 0};
    }

    //wait until all writes are finished and mark it.
    while (true) {
      if (leaf->key_array_rw_lock.increment_rd()) {break;}
    }
    int idx = leaf->find_key(key);
    
    if (idx < 0) {
      leaf->key_array_rw_lock.decrement_rd();
      rcu_progress(worker_id);
      return {false, 0};
    } else {
      P rval = leaf->get_payload(idx);
      leaf->key_array_rw_lock.decrement_rd();
      rcu_progress(worker_id);
      return {true, rval};
    }
  }

  // Looks for the last key no greater than the input value
  // Conceptually, this is equal to the last key before upper_bound()
  // returns end iterator on error
  // WARNING : iterator may cause error if other threads are also operating on ALEX
  // NOTE : the user should adequately call rcu_progress with thread_id for proper progress
  //        or use it when no other thread is working on ALEX.
  typename self_type::Iterator find_last_no_greater_than(const AlexKey<T>& key) {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {return end();}
    const int idx = leaf->upper_bound(key) - 1;
    if (idx >= 0) {
      return Iterator(leaf, idx);
    }

    // Edge case: need to check previous data node(s)
    while (true) {
      if (leaf->prev_leaf_.val_ == nullptr) {
        return Iterator(leaf, 0);
      }
      leaf = leaf->prev_leaf_.val_;
      if (leaf->num_keys_ > 0) {
        return Iterator(leaf, leaf->last_pos());
      }
    }
  }

  // Directly returns a pointer to the payload found through
  // find_last_no_greater_than(key)
  // This avoids the overhead of creating an iterator
  // returns nullptr on error.
  std::pair<bool, P> get_payload_last_no_greater_than(const AlexKey<T>& key, uint32_t worker_id) {
    stats_.num_lookups++;
    data_node_type* leaf = get_leaf(key, 0);
    if (leaf == nullptr) {
      rcu_progress(worker_id);
      return nullptr;
    }

    //wait until all wirtes are finished and mark it.
    while (true) {
      if (leaf->key_array_rw_lock.increment_rd()) {break;}
    }
    const int idx = leaf->upper_bound(key) - 1;
    if (idx >= 0) {
      P rval = leaf->get_payload(idx);
      leaf->key_array_rw_lock.decrement_rd();
      rcu_progress(worker_id);
      return {true, rval};
    }

    // Edge case: Need to check previous data node(s)
    while (true) {
      data_node_type *prev_leaf = leaf->prev_leaf_.read();
      if (prev_leaf == nullptr) {
        P rval = leaf->get_payload(leaf->first_pos());
        leaf->key_array_rw_lock.decrement_rd();
        rcu_progress(worker_id);
        return {true, rval};
      }
      leaf->key_array_rw_lock.decrement_rd();
      leaf = prev_leaf;
      if (leaf->num_keys_ > 0) {
        while (true) {
          if (leaf->key_array_rw_lock.increment_rd()) {break;}
        }
        P rval = leaf->get_payload(leaf->last_pos());
        leaf->key_array_rw_lock.decrement_rd();
        rcu_progress(worker_id);
        return {true, rval};
      }
    }
  }

  typename self_type::Iterator begin() {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return Iterator(static_cast<data_node_type*>(cur), 0);
  }

  typename self_type::Iterator end() {
    Iterator it = Iterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ConstIterator cbegin() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      cur = static_cast<model_node_type*>(cur)->children_[0];
    }
    return ConstIterator(static_cast<data_node_type*>(cur), 0);
  }

  typename self_type::ConstIterator cend() const {
    ConstIterator it = ConstIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ReverseIterator rbegin() {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto model_node = static_cast<model_node_type*>(cur);
      cur = model_node->children_[model_node->num_children_ - 1];
    }
    auto data_node = static_cast<data_node_type*>(cur);
    return ReverseIterator(data_node, data_node->data_capacity_ - 1);
  }

  typename self_type::ReverseIterator rend() {
    ReverseIterator it = ReverseIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  typename self_type::ConstReverseIterator crbegin() const {
    node_type* cur = root_node_;

    while (!cur->is_leaf_) {
      auto model_node = static_cast<model_node_type*>(cur);
      cur = model_node->children_[model_node->num_children_ - 1];
    }
    auto data_node = static_cast<data_node_type*>(cur);
    return ConstReverseIterator(data_node, data_node->data_capacity_ - 1);
  }

  typename self_type::ConstReverseIterator crend() const {
    ConstReverseIterator it = ConstReverseIterator();
    it.cur_leaf_ = nullptr;
    it.cur_idx_ = 0;
    return it;
  }

  /*** Insert ***/

 public:
  std::pair<Iterator, bool> insert(const V& value, uint32_t worker_id) {
    return insert(value.first, value.second, worker_id);
  }

  template <class InputIterator>
  void insert(InputIterator first, InputIterator last, uint32_t worker_id) {
    for (auto it = first; it != last; ++it) {
      insert(*it, worker_id);
    }
  }

  // This will NOT do an update of an existing key.
  // To perform an update or read-modify-write, do a lookup and modify the
  // payload's value.
  // Returns iterator to inserted element, and whether the insert happened or
  // not.
  // Insert does not happen if duplicates are not allowed and duplicate is
  // found.
  // If it failed finding a leaf, it returns iterator with null leaf with 0 index.
  // If it succeeded in finding a leaf, but failed because it's going to be destroyed
  // it returns iterator with null leaf with 1 index.
  std::pair<Iterator, bool> insert(const AlexKey<T>& key, const P& payload, uint32_t worker_id) {
    // in string ALEX, keys should not fall outside the key domain
    char larger_key = 0;
    char smaller_key = 0;
    for (unsigned int i = 0; i < key.max_key_length_; i++) {
      if (key.key_arr_[i] > istats_.key_domain_max_[i]) {larger_key = 1; break;}
      else if (key.key_arr_[i] < istats_.key_domain_min_[i]) {smaller_key = 1; break;}
    }
    if (larger_key || smaller_key) {
      std::cout << "worker ID : " << worker_id 
                << " root expansion should not happen." << std::endl;
      abort();
    }

    std::vector<TraversalNode<T, P>> traversal_path;
    data_node_type* leaf = get_leaf(key, worker_id, 1, &traversal_path);
    if (leaf == nullptr) {
      //failed finding leaf, shouldn't happen in normal cases.
      rcu_progress(worker_id);
      return {Iterator(nullptr, 0), false};
    } 
    
    leaf->unused.lock();
    memory_fence();
    if (leaf->unused.val_) { 
      //this leaf is about to be substituted.
      //should retry insert completely.
      leaf->unused.unlock();
      rcu_progress(worker_id);
      return {Iterator(nullptr, 1), false};
    }

    // Nonzero fail flag means that the insert did not happen
    std::pair<std::pair<int, int>, std::pair<data_node_type *, data_node_type *>> ret 
      = leaf->insert(key, payload, worker_id, &traversal_path);
    int fail = ret.first.first;
    int insert_pos = ret.first.second;
    leaf = ret.second.first;
    data_node_type *maybe_new_data_node = ret.second.second;

    if (fail == -1) {
      // Duplicate found and duplicates not allowed
      leaf->unused.unlock();
      memory_fence();
      if (maybe_new_data_node) { //new data node generated
        maybe_new_data_node->unused.unlock();
        memory_fence();
        rcu_barrier(worker_id);
        delete_node(leaf);
        delete  maybe_new_data_node->parent_->old_childrens_.at(leaf);
        maybe_new_data_node->parent_->old_childrens_.erase(leaf);
      }
      rcu_progress(worker_id);
      if (maybe_new_data_node) {return {Iterator(maybe_new_data_node, insert_pos), false};}
      else {return {Iterator(leaf, insert_pos), false};}
    }
    else if (!fail) {
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "alex.h insert : succeeded insertion and processing" << std::endl;
      alex::coutLock.unlock();
#endif
      //succeded in first try.
      leaf->unused.unlock();
      memory_fence();
      if (maybe_new_data_node) { //new data node generated
        maybe_new_data_node->unused.unlock();
        memory_fence();
        rcu_barrier(worker_id);
        delete_node(leaf);
        delete maybe_new_data_node->parent_->old_childrens_.at(leaf);
        maybe_new_data_node->parent_->old_childrens_.erase(leaf);
      }
      stats_.num_inserts.increment();
      stats_.num_keys.increment();
      rcu_progress(worker_id);
      if (maybe_new_data_node) {return {Iterator(maybe_new_data_node, insert_pos), true};}
      else {return {Iterator(leaf, insert_pos), true};}
    }
    // If no insert, and not duplicate,
    // figure out what to do with the data node to decrease the cost
    else {
      while (fail) {
        model_node_type* parent = leaf->parent_;
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "paernt is : " << parent << std::endl;
        alex::coutLock.unlock();
#endif
        leaf->unused.val_ = 1;
        auto start_time = std::chrono::high_resolution_clock::now();
        stats_.num_expand_and_scales.add(leaf->num_resizes_);

        int bucketID = traversal_path.back().bucketID;
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "bucketID : " << bucketID << std::endl;
        alex::coutLock.unlock();
#endif

        std::vector<fanout_tree::FTNode> used_fanout_tree_nodes;

        int fanout_tree_depth = 1;
        fanout_tree_depth = fanout_tree::find_best_fanout_existing_node<T, P>(
              leaf, stats_.num_keys.read(), used_fanout_tree_nodes, 2, worker_id);
              
        int best_fanout = 1 << fanout_tree_depth;
        stats_.cost_computation_time.add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count());

        if (fanout_tree_depth == 0) {
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "failed and decided to expand" << std::endl;
          alex::coutLock.unlock();
#endif
          // expand existing data node and retrain model
          data_node_type *resized_leaf = 
            leaf->resize(data_node_type::kMinDensity_, true,
                        leaf->is_append_mostly_right(),
                        leaf->is_append_mostly_left());
          resized_leaf->unused.lock();
          fanout_tree::FTNode& tree_node = used_fanout_tree_nodes[0];
          resized_leaf->cost_ = tree_node.cost;
          resized_leaf->expected_avg_exp_search_iterations_ =
              tree_node.expected_avg_search_iterations;
          resized_leaf->expected_avg_shifts_ = tree_node.expected_avg_shifts;
          resized_leaf->reset_stats();
          stats_.num_expand_and_retrains.increment();

          //substitute leaf pointer in parent model node
          int repeats = 1 << leaf->duplication_factor_;
          int start_bucketID = bucketID - (bucketID % repeats);
          int end_bucketID = start_bucketID + repeats;
          parent->children_.lock();
          node_type **parent_new_children = new (pointer_allocator().allocate(parent->num_children_))
            node_type *[parent->num_children_];
          node_type **parent_old_children = parent->children_.val_;
          std::copy(parent_old_children, parent_old_children + parent->num_children_,
                    parent_new_children);
          for (int i = start_bucketID; i < end_bucketID; i++) {
            parent_new_children[i] = resized_leaf;
          }
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "alex.h resizing children_" << std::endl;
          for (int i = 0; i < parent->num_children_; i++) {
            std::cout << i << " : " << parent_new_children[i] << std::endl;
          }
          alex::coutLock.unlock();
#endif
          parent->children_.val_ = parent_new_children;
          parent->children_.unlock();

          //wait before destruction of old leaf and metadata
          //Note that we don't let resized_leaf to be written by any other node yet.
          leaf->unused.unlock();
          memory_fence();
          rcu_barrier(worker_id);
          delete_node(leaf);
          delete parent_old_children;

          leaf = resized_leaf;
        } else {
          bool reuse_model = (fail == 3);
          // either split sideways or downwards
          // synchronization is covered automatically in splitting functions.
          bool should_split_downwards =
              (parent->num_children_ * best_fanout /
                       (1 << leaf->duplication_factor_) >
                   derived_params_.max_fanout ||
               parent->level_ == superroot_->level_ ||
               (fanout_tree_depth > leaf->duplication_factor_));
          if (should_split_downwards) {
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "failed and decided to split downwards" << std::endl;
            alex::coutLock.unlock();
#endif
            parent = split_downwards(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes,
                                     reuse_model, worker_id);
          } else {
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "failed and decided to split sideways" << std::endl;
            alex::coutLock.unlock();
#endif
            split_sideways(parent, bucketID, fanout_tree_depth, used_fanout_tree_nodes,
                           reuse_model, worker_id);
          }
          
          rcu_progress(worker_id);
          leaf = get_leaf(key, worker_id);
          leaf->unused.lock(); //note that this makes error if leaf is nullptr, or at get_leaf failure.
          memory_fence();
          if (leaf->unused.val_) { //should retry insert completely.
            leaf->unused.unlock();
            rcu_progress(worker_id);
            return {Iterator(nullptr, 1), false};
          }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = end_time - start_time;
        stats_.splitting_time.add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                .count());

        //empty used_fanout_tree_nodes for preventing memory leakage.
        for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {delete[] tree_node.a;}

        // Try again to insert the key
        traversal_path.clear();
        ret = leaf->insert(key, payload, worker_id, &traversal_path);
        fail = ret.first.first;
        insert_pos = ret.first.second;
        leaf = ret.second.first;
        maybe_new_data_node = ret.second.second;
        if (fail == -1) {
          // Duplicate found and duplicates not allowed
          leaf->unused.unlock();
          memory_fence();
          if (maybe_new_data_node) { //new data node generated
            maybe_new_data_node->unused.unlock();
            memory_fence();
            rcu_barrier(worker_id);
            delete_node(leaf);
            delete parent->old_childrens_.at(leaf);
            parent->old_childrens_.erase(leaf);
          }
          rcu_progress(worker_id);
          if (maybe_new_data_node) {return {Iterator(maybe_new_data_node, insert_pos), false};}
          else {return {Iterator(leaf, insert_pos), false};}
        }
      }
      leaf->unused.unlock();
      memory_fence();
      if (maybe_new_data_node) { //new data node generated
        maybe_new_data_node->unused.unlock();
        memory_fence();
        rcu_barrier(worker_id);
        delete_node(leaf);
        delete maybe_new_data_node->parent_->old_childrens_.at(leaf);
        maybe_new_data_node->parent_->old_childrens_.erase(leaf);
      }
      stats_.num_inserts.increment();
      stats_.num_keys.increment();
      rcu_progress(worker_id);
      if (maybe_new_data_node) {return {Iterator(maybe_new_data_node, insert_pos), true};}
      else {return {Iterator(leaf, insert_pos), true};}
    }
  }

 private:

  // Splits downwards in the manner determined by the fanout tree and updates
  // the pointers of the parent.
  // If no fanout tree is provided, then splits downward in two. Returns the
  // newly created model node.
  model_node_type* split_downwards(
      model_node_type* parent, int bucketID, int fanout_tree_depth,
      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
      bool reuse_model, uint32_t worker_id) {
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "...bucketID : " << bucketID << std::endl;
    alex::coutLock.unlock();
#endif
    node_type **parent_children_ = parent->children_.read();
    auto leaf = static_cast<data_node_type*> (parent_children_[bucketID]);
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "and leaf is : " << leaf << std::endl;
    alex::coutLock.unlock();
#endif
    stats_.num_downward_splits.increment();
    stats_.num_downward_split_keys.add(leaf->num_keys_);

    // Create the new model node that will replace the current data node
    int fanout = 1 << fanout_tree_depth;
    auto new_node = new (model_node_allocator().allocate(1))
        model_node_type(leaf->level_, parent, max_key_length_, allocator_);
    new_node->duplication_factor_ = leaf->duplication_factor_;
    new_node->num_children_ = fanout;
    new_node->children_.val_ =
        new (pointer_allocator().allocate(fanout)) node_type*[fanout];
    //needs to initialize min/max key in case of split_downwards.
    std::copy(leaf->min_key_.val_->key_arr_, leaf->min_key_.val_->key_arr_ + max_key_length_,
              new_node->min_key_.val_->key_arr_);
    std::copy(leaf->max_key_.val_->key_arr_, leaf->max_key_.val_->key_arr_ + max_key_length_,
              new_node->max_key_.val_->key_arr_);


    int repeats = 1 << leaf->duplication_factor_;
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
    int end_bucketID =
        start_bucketID + repeats;  // first bucket with different child

    //make an iterator, and add them all into the model
    //and train to make it have an output of [0, fanout]
    LinearModel<T> tmp_model(max_key_length_);
    LinearModelBuilder<T> tmp_model_builder(&tmp_model);
    Iterator it(leaf, 0);

    int key_cnt = 0;
    while(it.cur_idx_ != -1) {
      //std::cout << it.key().key_arr_ << std::endl;
      tmp_model_builder.add(it.key(), ((double) key_cnt * fanout / (leaf->num_keys_ - 1)));
      key_cnt++;
      it++;
      if (it.cur_leaf_ != leaf) {break;} //moved out to next node.
    }
    tmp_model_builder.build();
    std::copy(tmp_model.a_, tmp_model.a_ + tmp_model.max_key_length_, new_node->model_.a_);
    new_node->model_.b_ = tmp_model.b_;

#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "left prediction result (sd) " << new_node->model_.predict_double(leaf->key_slots_[leaf->first_pos()]) << std::endl;
    std::cout << "right prediction result (sd) " << new_node->model_.predict_double(leaf->key_slots_[leaf->last_pos()]) << std::endl;
    alex::coutLock.unlock();
#endif
    node_type **switched_children;

    // Create new data nodes
    if (used_fanout_tree_nodes.empty()) {
      assert(fanout_tree_depth == 1);
      switched_children = create_two_new_data_nodes(leaf, new_node, fanout_tree_depth,
                                                    reuse_model, worker_id);
    } else {
      switched_children = create_new_data_nodes(leaf, new_node, fanout_tree_depth,
                            used_fanout_tree_nodes, worker_id);
    }

    stats_.num_data_nodes.decrement();
    stats_.num_model_nodes.increment();

    //substitute pointers in parent model node
    parent->children_.lock();
    node_type **parent_new_children = new (pointer_allocator().allocate(parent->num_children_))
            node_type *[parent->num_children_];
    node_type **parent_old_children = parent->children_.val_;
    std::copy(parent_old_children, parent_old_children + parent->num_children_,
              parent_new_children);
    for (int i = start_bucketID; i < end_bucketID; i++) {
      parent_new_children[i] = new_node;
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "split_downwards parent children_" << std::endl;
    for (int i = 0; i < parent->num_children_; i++) {
      std::cout << i << " : " << parent_new_children[i] << std::endl;
    }
    alex::coutLock.unlock();
#endif
    parent->children_.val_ = parent_new_children;
    parent->children_.unlock();
    if (parent == superroot_) {
      root_node_ = new_node;
    }

    //destroy unused leaf and meta data after all threads finished using.
    leaf->unused.val_ = 1;
    memory_fence();
    leaf->unused.unlock();
    memory_fence();
    rcu_barrier(worker_id);
    delete_node(leaf);
    delete parent_old_children;
    delete switched_children;

#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "min_key_(model_node) : " << new_node->min_key_.val_->key_arr_ << std::endl;
    std::cout << "max_key_(model_node) : " << new_node->max_key_.val_->key_arr_ << std::endl;
    for (int i = 0; i < fanout; i++) {
        std::cout << i << "'s min_key is : "
                  << new_node->children_.val_[i]->min_key_.val_->key_arr_ << std::endl;
        std::cout << i << "'s max_key is : " 
                  << new_node->children_.val_[i]->max_key_.val_->key_arr_ << std::endl;
    }
    alex::coutLock.unlock();
#endif

    return new_node;
  }

  // Splits data node sideways in the manner determined by the fanout tree.
  // If no fanout tree is provided, then splits sideways in two.
  void split_sideways(model_node_type* parent, int bucketID,
                      int fanout_tree_depth,
                      std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
                      bool reuse_model, uint32_t worker_id) {
    auto leaf = static_cast<data_node_type*>((parent->children_.read())[bucketID]);
    stats_.num_sideways_splits.increment();
    stats_.num_sideways_split_keys.add(leaf->num_keys_);

    int fanout = 1 << fanout_tree_depth;
    int repeats = 1 << leaf->duplication_factor_;
    if (fanout > repeats) {
      //in multithreading, because of synchronization issue of duplication_fcator_
      //we don't do model expansion.
      ;
    }
    int start_bucketID =
        bucketID - (bucketID % repeats);  // first bucket with same child
    
    node_type **parent_old_children;

    if (used_fanout_tree_nodes.empty()) {
      assert(fanout_tree_depth == 1);
      parent_old_children = create_two_new_data_nodes(
          leaf, parent,
          std::max(fanout_tree_depth,
                   static_cast<int>(leaf->duplication_factor_)),
          reuse_model, worker_id, start_bucketID);
    } else {
      // Extra duplication factor is required when there are more redundant
      // pointers than necessary
      int extra_duplication_factor =
          std::max(0, leaf->duplication_factor_ - fanout_tree_depth);
      parent_old_children = create_new_data_nodes(leaf, parent, fanout_tree_depth,
                            used_fanout_tree_nodes, worker_id, 
                            start_bucketID, extra_duplication_factor);
    }

    leaf->unused.val_ = 1;
    memory_fence();
    leaf->unused.unlock();
    memory_fence();
    rcu_barrier(worker_id);
    delete_node(leaf);
    delete parent_old_children;

    stats_.num_data_nodes.decrement();
  }

  // Create two new data nodes by equally dividing the key space of the old data
  // node, insert the new
  // nodes as children of the parent model node starting from a given position,
  // and link the new data nodes together.
  // duplication_factor denotes how many child pointer slots were assigned to
  // the old data node.
  // returns destroy needed old meta data.
  node_type ** create_two_new_data_nodes(data_node_type* old_node,
                                 model_node_type* parent,
                                 int duplication_factor, bool reuse_model,
                                 uint32_t worker_id, int start_bucketID = 0) {
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "called create_two_new_dn" << std::endl;
    alex::coutLock.unlock();
#endif
    assert(duplication_factor >= 1);
    int num_buckets = 1 << duplication_factor;
    int end_bucketID = start_bucketID + num_buckets;
    int mid_bucketID = start_bucketID + num_buckets / 2;

    bool append_mostly_right = old_node->is_append_mostly_right();
    int appending_right_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->max_key_.val_)), 0),
        parent->num_children_ - 1);
    bool append_mostly_left = old_node->is_append_mostly_left();
    int appending_left_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->min_key_.val_)), 0),
        parent->num_children_ - 1);

    int right_boundary = 0;
    AlexKey<T> tmpkey = AlexKey<T>(max_key_length_);
    //According to my insight, linear model would be monotonically increasing
    //So I think we could compute key corresponding to mid_bucketID as
    //average of min/max key of current splitting node.
    for (unsigned int i = 0; i < max_key_length_; i++) {
      tmpkey.key_arr_[i] = (old_node->max_key_.val_->key_arr_[i] + old_node->min_key_.val_->key_arr_[i]) / 2;
    }
    
    right_boundary = old_node->lower_bound(tmpkey);
    // Account for off-by-one errors due to floating-point precision issues.
    while (right_boundary < old_node->data_capacity_) {
      AlexKey<T> old_rbkey = old_node->get_key(right_boundary);
      if (key_equal(old_rbkey, old_node->kEndSentinel_)) {break;}
      if (parent->model_.predict(old_node->get_key(right_boundary)) >= mid_bucketID) {break;}
      right_boundary = std::min(
          old_node->get_next_filled_position(right_boundary, false) + 1,
          old_node->data_capacity_);
    }
    data_node_type* left_leaf = bulk_load_leaf_node_from_existing(
        old_node, 0, right_boundary, worker_id, true, nullptr, reuse_model,
        append_mostly_right && start_bucketID <= appending_right_bucketID &&
            appending_right_bucketID < mid_bucketID,
        append_mostly_left && start_bucketID <= appending_left_bucketID &&
            appending_left_bucketID < mid_bucketID);
    data_node_type* right_leaf = bulk_load_leaf_node_from_existing(
        old_node, right_boundary, old_node->data_capacity_, worker_id, true, nullptr, reuse_model,
        append_mostly_right && mid_bucketID <= appending_right_bucketID &&
            appending_right_bucketID < end_bucketID,
        append_mostly_left && mid_bucketID <= appending_left_bucketID &&
            appending_left_bucketID < end_bucketID);
    old_node->pending_left_leaf_ = left_leaf;
    old_node->pending_right_leaf_ = right_leaf;
    left_leaf->level_ = static_cast<short>(parent->level_ + 1);
    right_leaf->level_ = static_cast<short>(parent->level_ + 1);
    left_leaf->duplication_factor_ =
        static_cast<uint8_t>(duplication_factor - 1);
    right_leaf->duplication_factor_ =
        static_cast<uint8_t>(duplication_factor - 1);
    left_leaf->parent_ = parent;
    right_leaf->parent_ = parent;

    parent->children_.lock();
    node_type **parent_new_children = new (pointer_allocator().allocate(parent->num_children_))
      node_type*[parent->num_children_];
    node_type **parent_old_children = parent->children_.val_;
    std::copy(parent_old_children, parent_old_children + parent->num_children_,
              parent_new_children);
    for (int i = start_bucketID; i < mid_bucketID; i++) {
      parent_new_children[i] = left_leaf;
    }
    for (int i = mid_bucketID; i < end_bucketID; i++) {
      parent_new_children[i] = right_leaf;
    }
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "two new data node made with left min/max as "
                << left_leaf->min_key_.val_->key_arr_
                << " " << left_leaf->max_key_.val_->key_arr_
                << "and right min/max as "
                << right_leaf->min_key_.val_->key_arr_
                << " " << right_leaf->max_key_.val_->key_arr_
                << std::endl;
      alex::coutLock.unlock();
#endif
    parent->children_.val_ = parent_new_children;
    parent->children_.unlock();
    link_data_nodes(old_node, left_leaf, right_leaf);
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "finished create_two_new_dn" << std::endl;
    alex::coutLock.unlock();
#endif

    return parent_old_children;
  }

  // Create new data nodes from the keys in the old data node according to the
  // fanout tree, insert the new
  // nodes as children of the parent model node starting from a given position,
  // and link the new data nodes together.
  // Helper for splitting when using a fanout tree.
  // returns destroy needed old meta data.
 node_type **create_new_data_nodes(
      data_node_type* old_node, model_node_type* parent,
      int fanout_tree_depth, std::vector<fanout_tree::FTNode>& used_fanout_tree_nodes,
      uint32_t worker_id, int start_bucketID = 0, int extra_duplication_factor = 0) {
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "called create_new_dn" << std::endl;
    std::cout << "old node is " << old_node << std::endl;
    alex::coutLock.unlock();
#endif
    bool append_mostly_right = old_node->is_append_mostly_right();
    int appending_right_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->max_key_.val_)), 0),
        parent->num_children_ - 1);
    bool append_mostly_left = old_node->is_append_mostly_left();
    int appending_left_bucketID = std::min<int>(
        std::max<int>(parent->model_.predict(*(old_node->min_key_.val_)), 0),
        parent->num_children_ - 1);

    // Create the new data nodes
    int cur = start_bucketID;  // first bucket with same child
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << start_bucketID << std::endl;
    alex::coutLock.unlock();
#endif
    data_node_type* prev_leaf =
        old_node->prev_leaf_.read();  // used for linking the new data nodes
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "initial prev_leaf is : " << prev_leaf << std::endl;
    alex::coutLock.unlock();
#endif
    int left_boundary = 0;
    int right_boundary = 0;
    // Keys may be re-assigned to an adjacent fanout tree node due to off-by-one
    // errors
    int num_reassigned_keys = 0;
    int first_iter = 1;
    parent->children_.lock();
    node_type **parent_old_children = parent->children_.val_;
    node_type **parent_new_children = new (pointer_allocator().allocate(parent->num_children_))
      node_type*[parent->num_children_];
    std::copy(parent_old_children, parent_old_children + parent->num_children_,
              parent_new_children);
    for (fanout_tree::FTNode& tree_node : used_fanout_tree_nodes) {
      left_boundary = right_boundary;
      auto duplication_factor = static_cast<uint8_t>(
          fanout_tree_depth - tree_node.level + extra_duplication_factor);
      int child_node_repeats = 1 << duplication_factor;
      bool keep_left = append_mostly_right && cur <= appending_right_bucketID &&
                       appending_right_bucketID < cur + child_node_repeats;
      bool keep_right = append_mostly_left && cur <= appending_left_bucketID &&
                        appending_left_bucketID < cur + child_node_repeats;
      right_boundary = tree_node.right_boundary;
      // Account for off-by-one errors due to floating-point precision issues.
      tree_node.num_keys -= num_reassigned_keys;
      num_reassigned_keys = 0;
      while (right_boundary < old_node->data_capacity_) {
        AlexKey<T> old_node_rbkey = old_node->get_key(right_boundary);
        if (key_equal(old_node_rbkey, old_node->kEndSentinel_)) {break;}
        if (parent->model_.predict(old_node->get_key(right_boundary)) >=
                 cur + child_node_repeats) {break;}
        num_reassigned_keys++;
        right_boundary = std::min(
            old_node->get_next_filled_position(right_boundary, false) + 1,
            old_node->data_capacity_);
      }
      tree_node.num_keys += num_reassigned_keys;
      data_node_type* child_node = bulk_load_leaf_node_from_existing(
          old_node, left_boundary, right_boundary, worker_id, false, &tree_node, false,
          keep_left, keep_right);
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "child_node pointer : " << child_node << std::endl;
      alex::coutLock.unlock();
#endif
      child_node->level_ = static_cast<short>(parent->level_ + 1);
      child_node->cost_ = tree_node.cost;
      child_node->duplication_factor_ = duplication_factor;
      child_node->expected_avg_exp_search_iterations_ =
          tree_node.expected_avg_search_iterations;
      child_node->expected_avg_shifts_ = tree_node.expected_avg_shifts;

      if (first_iter) { //left leaf is not a new data node
        old_node->pending_left_leaf_.update(child_node);
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "updated pll with " << child_node << std::endl;
        alex::coutLock.unlock();
#endif
        if (prev_leaf != nullptr) {
          data_node_type *prev_leaf_pending_rl = prev_leaf->pending_right_leaf_.read();
          if (prev_leaf_pending_rl != nullptr) {
            child_node->prev_leaf_.update(prev_leaf_pending_rl);
            prev_leaf_pending_rl->next_leaf_.update(child_node);
          }
          else {
#if DEBUG_PRINT
            alex::coutLock.lock();
            std::cout << "t" << worker_id << " - ";
            std::cout << "child_node's prev_leaf_ is " << prev_leaf << std::endl;
            alex::coutLock.unlock();
#endif
            child_node->prev_leaf_.update(prev_leaf);
            prev_leaf->next_leaf_.update(child_node);
          }
        }
        else {
          child_node->prev_leaf_.update(nullptr);
        }
        first_iter = 0;
      }
      else {
#if DEBUG_PRINT
        alex::coutLock.lock();
        std::cout << "t" << worker_id << " - ";
        std::cout << "child_node's prev_leaf_ is " << prev_leaf << std::endl;
        alex::coutLock.unlock();
#endif
        child_node->prev_leaf_.update(prev_leaf);
        prev_leaf->next_leaf_.update(child_node);
      }
      child_node->parent_ = parent;

      //update model node metadata
      for (int i = cur; i < cur + child_node_repeats; i++) {
        parent_new_children[i] = child_node;
      }
      cur += child_node_repeats;
      prev_leaf = child_node;
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "new data node made with min_key as "
                << child_node->min_key_.val_->key_arr_
                << " and max_key as "
                << child_node->max_key_.val_->key_arr_
                << std::endl;
      alex::coutLock.unlock();
#endif
    }
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "cndn children_" << std::endl;
      for (int i = 0 ; i < parent->num_children_; i++) {
        std::cout << i << " : " << parent_new_children[i] << std::endl;
      }
      alex::coutLock.unlock();
#endif
    parent->children_.val_ = parent_new_children;
    parent->children_.unlock();

    //update right-most leaf's next/prev leaf.
    old_node->pending_right_leaf_.update(prev_leaf);
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "updated prl with " << prev_leaf << std::endl;
    alex::coutLock.unlock();
#endif
    data_node_type *next_leaf = old_node->next_leaf_.read();
    if (next_leaf != nullptr) {
      data_node_type *next_leaf_pending_ll = next_leaf->pending_left_leaf_.read();
      if (next_leaf_pending_ll != nullptr) {
        prev_leaf->next_leaf_.update(next_leaf_pending_ll);
        next_leaf_pending_ll->prev_leaf_.update(prev_leaf);
      }
      else {
        prev_leaf->next_leaf_.update(next_leaf);
        next_leaf->prev_leaf_.update(prev_leaf);
      }
    }
    else {
      prev_leaf->next_leaf_.update(nullptr);
    }
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "finished create_new_dn" << std::endl;
    alex::coutLock.unlock();
#endif
    return parent_old_children; 
  }

  /*** Stats ***/

 public:
  // Number of elements
  size_t size() const { return static_cast<size_t>(stats_.num_keys.read()); }

  // True if there are no elements
  bool empty() const { return (size() == 0); }

  // This is just a function required by the STL standard. ALEX can hold more
  // items.
  size_t max_size() const { return size_t(-1); }

  // Size in bytes of all the keys, payloads, and bitmaps stored in this index
  long long data_size() const {
    long long size = 0;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      node_type* cur = node_it.current();
      if (cur->is_leaf_) {
        size += static_cast<data_node_type*>(cur)->data_size();
      }
    }
    return size;
  }

  // Size in bytes of all the model nodes (including pointers) and metadata in
  // data nodes
  // should only be called when alex structure is not being modified.
  long long model_size() const {
    long long size = 0;
    for (NodeIterator node_it = NodeIterator(this); !node_it.is_end();
         node_it.next()) {
      size += node_it.current()->node_size();
    }
    return size;
  }

  // Total number of nodes in the RMI
  int num_nodes() const {
    return stats_.num_data_nodes.read() + stats_.num_model_nodes.read();
  };

  // Number of data nodes in the RMI
  int num_leaves() const { return stats_.num_data_nodes.read(); };

  // Return a const reference to the current statistics
  const struct Stats& get_stats() const { return stats_; }

  /*** Debugging ***/

 public:
  // If short_circuit is true, then we stop validating after detecting the first
  // invalid property
  bool validate_structure(bool validate_data_nodes = false,
                          bool short_circuit = false) const {
    bool is_valid = true;
    std::stack<node_type*> node_stack;
    node_type* cur;
    node_stack.push(root_node_);

    while (!node_stack.empty()) {
      cur = node_stack.top();
      node_stack.pop();

      if (!cur->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur);
        if (!node->validate_structure(true)) {
          std::cout << "[Model node invalid structure]"
                    << " node addr: " << node
                    << ", node level: " << node->level_ << std::endl;
          if (short_circuit) {
            return false;
          } else {
            is_valid = false;
          }
        }

        node_stack.push(node->children_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_[i] != node->children_[i + 1]) {
            node_stack.push(node->children_[i]);
          }
        }
      } else {
        if (validate_data_nodes) {
          auto node = static_cast<data_node_type*>(cur);
          if (!node->validate_structure(true)) {
            std::cout << "[Data node invalid structure]"
                      << " node addr: " << node
                      << ", node level: " << node->level_ << std::endl;
            if (short_circuit) {
              return false;
            } else {
              is_valid = false;
            }
          }
          if (node->num_keys_ > 0) {
            data_node_type* prev_nonempty_leaf = node->prev_leaf_.read();
            while (prev_nonempty_leaf != nullptr &&
                   prev_nonempty_leaf->num_keys_ == 0) {
              prev_nonempty_leaf = prev_nonempty_leaf->prev_leaf_.read();;
            }
            if (prev_nonempty_leaf) {
              AlexKey<T> last_in_prev_leaf = prev_nonempty_leaf->last_key();
              AlexKey<T> first_in_cur_leaf = node->first_key();
              if (!Compare(last_in_prev_leaf, first_in_cur_leaf)) {
                std::cout
                    << "[Data node keys not in sorted order with prev node]"
                    << " node addr: " << node
                    << ", node level: " << node->level_
                    << ", last in prev leaf: " << last_in_prev_leaf
                    << ", first in cur leaf: " << first_in_cur_leaf
                    << std::endl;
                if (short_circuit) {
                  return false;
                } else {
                  is_valid = false;
                }
              }
            }
            data_node_type* next_nonempty_leaf = node->next_leaf_;
            while (next_nonempty_leaf != nullptr &&
                   next_nonempty_leaf->num_keys_ == 0) {
              next_nonempty_leaf = next_nonempty_leaf->next_leaf_;
            }
            if (next_nonempty_leaf) {
              AlexKey<T> first_in_next_leaf = next_nonempty_leaf->first_key();
              AlexKey<T> last_in_cur_leaf = node->last_key();
              if (!Compare(last_in_cur_leaf, first_in_next_leaf)) {
                std::cout
                    << "[Data node keys not in sorted order with next node]"
                    << " node addr: " << node
                    << ", node level: " << node->level_
                    << ", last in cur leaf: " << last_in_cur_leaf
                    << ", first in next leaf: " << first_in_next_leaf
                    << std::endl;
                if (short_circuit) {
                  return false;
                } else {
                  is_valid = false;
                }
              }
            }
          }
        }
      }
    }
    return is_valid;
  }

  /*** Iterators ***/

 public:
  class Iterator {
   public:
    data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    Iterator() {}

    Iterator(data_node_type* leaf, int idx) : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    Iterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    Iterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    Iterator& operator=(const Iterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    Iterator& operator++() {
      advance();
      return *this;
    }

    Iterator operator++(int) {
      Iterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_].val_);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const {return ((data_node_type *) cur_leaf_)->get_key(cur_idx_); }

    P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_.read();
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
          cur_leaf_ = cur_leaf_->next_leaf_.read();
          cur_idx_ = 0;
          if (cur_leaf_ == nullptr) {
            return;
          }
          cur_bitmap_idx_ = 0;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }
  };

  class ConstIterator {
   public:
    const data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ConstIterator() {}

    ConstIterator(const data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ConstIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstIterator(const ConstIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstIterator(const ConstReverseIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstIterator& operator=(const ConstIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ConstIterator& operator++() {
      advance();
      return *this;
    }

    ConstIterator operator++(int) {
      ConstIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    const V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    const P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ConstIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ConstIterator& rhs) const { return !(*this == rhs); };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= cur_leaf_->bitmap_size_) {
          cur_leaf_ = cur_leaf_->next_leaf_;
          cur_idx_ = 0;
          if (cur_leaf_ == nullptr) {
            return;
          }
          cur_bitmap_idx_ = 0;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }
  };

  class ReverseIterator {
   public:
    data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ReverseIterator() {}

    ReverseIterator(data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ReverseIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ReverseIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ReverseIterator& operator=(const ReverseIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ReverseIterator& operator++() {
      advance();
      return *this;
    }

    ReverseIterator operator++(int) {
      ReverseIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ReverseIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ReverseIterator& rhs) const {
      return !(*this == rhs);
    };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);

      advance();
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_--;
        if (cur_bitmap_idx_ < 0) {
          cur_leaf_ = cur_leaf_->prev_leaf_.read();
          if (cur_leaf_ == nullptr) {
            cur_idx_ = 0;
            return;
          }
          cur_idx_ = cur_leaf_->data_capacity_ - 1;
          cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      assert(cpu_supports_bmi());
      int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
      cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
      cur_bitmap_data_ &= ~(1ULL << bit_pos);
    }
  };

  class ConstReverseIterator {
   public:
    const data_node_type* cur_leaf_ = nullptr;  // current data node
    int cur_idx_ = 0;         // current position in key/data_slots of data node
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ = 0;  // caches the relevant data in the current
                                    // bitmap position

    ConstReverseIterator() {}

    ConstReverseIterator(const data_node_type* leaf, int idx)
        : cur_leaf_(leaf), cur_idx_(idx) {
      initialize();
    }

    ConstReverseIterator(const ConstReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstReverseIterator(const ReverseIterator& other)
        : cur_leaf_(other.cur_leaf_),
          cur_idx_(other.cur_idx_),
          cur_bitmap_idx_(other.cur_bitmap_idx_),
          cur_bitmap_data_(other.cur_bitmap_data_) {}

    ConstReverseIterator(const Iterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstReverseIterator(const ConstIterator& other)
        : cur_leaf_(other.cur_leaf_), cur_idx_(other.cur_idx_) {
      initialize();
    }

    ConstReverseIterator& operator=(const ConstReverseIterator& other) {
      if (this != &other) {
        cur_idx_ = other.cur_idx_;
        cur_leaf_ = other.cur_leaf_;
        cur_bitmap_idx_ = other.cur_bitmap_idx_;
        cur_bitmap_data_ = other.cur_bitmap_data_;
      }
      return *this;
    }

    ConstReverseIterator& operator++() {
      advance();
      return *this;
    }

    ConstReverseIterator operator++(int) {
      ConstReverseIterator tmp = *this;
      advance();
      return tmp;
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    // Does not return a reference because keys and payloads are stored
    // separately.
    // If possible, use key() and payload() instead.
    V operator*() const {
      return std::make_pair(cur_leaf_->key_slots_[cur_idx_],
                            cur_leaf_->payload_slots_[cur_idx_]);
    }
#else
    // If data node stores key-payload pairs contiguously, return reference to V
    const V& operator*() const { return cur_leaf_->data_slots_[cur_idx_]; }
#endif

    const AlexKey<T>& key() const { return cur_leaf_->get_key(cur_idx_); }

    const P& payload() const { return cur_leaf_->get_payload(cur_idx_); }

    bool is_end() const { return cur_leaf_ == nullptr; }

    bool operator==(const ConstReverseIterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_ && cur_leaf_ == rhs.cur_leaf_;
    }

    bool operator!=(const ConstReverseIterator& rhs) const {
      return !(*this == rhs);
    };

   private:
    void initialize() {
      if (!cur_leaf_) return;
      assert(cur_idx_ >= 0);
      if (cur_idx_ >= cur_leaf_->data_capacity_) {
        cur_leaf_ = cur_leaf_->next_leaf_;
        cur_idx_ = 0;
        if (!cur_leaf_) return;
      }

      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= (1ULL << bit_pos) | ((1ULL << bit_pos) - 1);

      advance();
    }

    forceinline void advance() {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_--;
        if (cur_bitmap_idx_ < 0) {
          cur_leaf_ = cur_leaf_->prev_leaf_.read();
          if (cur_leaf_ == nullptr) {
            cur_idx_ = 0;
            return;
          }
          cur_idx_ = cur_leaf_->data_capacity_ - 1;
          cur_bitmap_idx_ = cur_leaf_->bitmap_size_ - 1;
        }
        cur_bitmap_data_ = cur_leaf_->bitmap_[cur_bitmap_idx_];
      }
      assert(cpu_supports_bmi());
      int bit_pos = static_cast<int>(63 - _lzcnt_u64(cur_bitmap_data_));
      cur_idx_ = (cur_bitmap_idx_ << 6) + bit_pos;
      cur_bitmap_data_ &= ~(1ULL << bit_pos);
    }
  };

  // Iterates through all nodes with pre-order traversal
  class NodeIterator {
   public:
    const self_type* index_;
    node_type* cur_node_;
    std::stack<node_type*> node_stack_;  // helps with traversal

    // Start with root as cur and all children of root in stack
    explicit NodeIterator(const self_type* index)
        : index_(index), cur_node_(index->root_node_) {
      if (cur_node_ && !cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_.val_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_.val_[i] != node->children_.val_[i + 1]) {
            node_stack_.push(node->children_.val_[i]);
          }
        }
      }
    }

    node_type* current() const { return cur_node_; }

    node_type* next() {
      if (node_stack_.empty()) {
        cur_node_ = nullptr;
        return nullptr;
      }

      cur_node_ = node_stack_.top();
      node_stack_.pop();

      if (!cur_node_->is_leaf_) {
        auto node = static_cast<model_node_type*>(cur_node_);
        node_stack_.push(node->children_.val_[node->num_children_ - 1]);
        for (int i = node->num_children_ - 2; i >= 0; i--) {
          if (node->children_.val_[i] != node->children_.val_[i + 1]) {
            node_stack_.push(node->children_.val_[i]);
          }
        }
      }

      return cur_node_;
    }

    bool is_end() const { return cur_node_ == nullptr; }
  };
};
}  // namespace alex
