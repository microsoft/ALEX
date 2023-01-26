// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/*
 * This file contains code for ALEX nodes. There are two types of nodes in ALEX:
 * - Model nodes (equivalent to internal/inner nodes of a B+ Tree)
 * - Data nodes, sometimes referred to as leaf nodes (equivalent to leaf nodes
 * of a B+ Tree)
 */

#pragma once

#include "alex_base.h"

// Whether we store key and payload arrays separately in data nodes
// By default, we store them separately
#define ALEX_DATA_NODE_SEP_ARRAYS 1

#if ALEX_DATA_NODE_SEP_ARRAYS
#define ALEX_DATA_NODE_KEY_AT(i) key_slots_[i]
#define ALEX_DATA_NODE_PAYLOAD_AT(i) payload_slots_[i]
#else
#define ALEX_DATA_NODE_KEY_AT(i) data_slots_[i].first
#define ALEX_DATA_NODE_PAYLOAD_AT(i) data_slots_[i].second
#endif

// Whether we use lzcnt and tzcnt when manipulating a bitmap (e.g., when finding
// the closest gap).
// If your hardware does not support lzcnt/tzcnt (e.g., your Intel CPU is
// pre-Haswell), set this to 0.
#define ALEX_USE_LZCNT 1

namespace alex {

//forward declaration.
template <class P, class Alloc> class AlexModelNode;

// A parent class for both types of ALEX nodes
template <class P, class Alloc = std::allocator<std::pair<AlexKey, P>>>
class AlexNode {
 public:

  typedef AlexNode<P> self_type;
  typedef AlexModelNode<P, Alloc> model_node_type;

  // Whether this node is a leaf (data) node
  bool is_leaf_ = false;

  // Power of 2 to which the pointer to this node is duplicated in its parent
  // model node
  // For example, if duplication_factor_ is 3, then there are 8 redundant
  // pointers to this node in its parent
  uint8_t duplication_factor_ = 0;

  // Node's level in the RMI. Root node is level 0
  short level_ = 0;

  // Both model nodes and data nodes nodes use models
  LinearModel model_ = LinearModel(1);

  // Node's linear model's key's max_length.
  unsigned int max_key_length_ = 1;

  // Could be either the expected or empirical cost, depending on how this field
  // is used
  double cost_ = 0.0;

  //parent of current node. Root is nullptr. Need to be given by parameter.
  model_node_type *parent_ = nullptr;

  AlexNode() = default;
  explicit AlexNode(short level) : level_(level) {}
  AlexNode(short level, bool is_leaf) : is_leaf_(is_leaf), level_(level) {}
  AlexNode(short level, bool is_leaf, self_type *parent)
      : level_(level), is_leaf_(is_leaf), parent_(parent) {}
  AlexNode(short level, bool is_leaf, 
      model_node_type *parent, unsigned int max_key_length) 
        : is_leaf_(is_leaf), level_(level), model_(max_key_length), 
          max_key_length_(max_key_length), parent_(parent) {}

  AlexNode(const self_type& other)
      : is_leaf_(other.is_leaf_),
        duplication_factor_(other.duplication_factor_),
        level_(other.level_),
        max_key_length_(other.max_key_length_),
        cost_(other.cost_),
        parent_(other.parent_) {
    model_ = LinearModel(other.max_key_length_);
    if (other.model_.a_ != nullptr) {
      model_.a_ = new double[max_key_length_]();
      for (int i = 0; i < max_key_length_; i++) {
        model_.a_[i] = other.model_.a_[i];
      }
    }
    model_.b_ = other.model_.b_;
  }
  virtual ~AlexNode() = default;

  // The size in bytes of all member variables in this class
  virtual long long node_size() const = 0;
};

template <class P, class Alloc = std::allocator<std::pair<AlexKey, P>>>
class AlexModelNode : public AlexNode<P> {
 public:
  typedef AlexNode<P> basic_node_type;
  typedef AlexModelNode<P, Alloc> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexNode<P>*>::other
      pointer_alloc_type;

  const Alloc& allocator_;

  // Number of logical children. Must be a power of 2
  int num_children_ = 0;

  // Array of pointers to children
  AlexNode<P>** children_ = nullptr;

  // Minimum / Maximum key's data in node
  double *Mnode_min_key_ = nullptr;
  double *Mnode_max_key_ = nullptr;

  explicit AlexModelNode(const Alloc& alloc = Alloc())
      : AlexNode<P>(0, false), allocator_(alloc) {}

  explicit AlexModelNode(short level, const Alloc& alloc = Alloc())
      : AlexNode<P>(level, false), allocator_(alloc) {}

  explicit AlexModelNode(short level, basic_node_type *parent,
                         const Alloc& alloc = Alloc())
      : AlexNode<P>(level, false, parent), allocator_(alloc){}
  
  explicit AlexModelNode(short level, self_type *parent,
                         unsigned int max_key_length, const Alloc& alloc = Alloc())
      : AlexNode<P>(level, false, parent, max_key_length), allocator_(alloc) {}

  ~AlexModelNode() {
    if (children_ != nullptr) {
      pointer_allocator().deallocate(children_, num_children_);
    }
    if (Mnode_min_key_ != nullptr) {
      delete[] Mnode_min_key_;
    }
    if (Mnode_max_key_ != nullptr) {
      delete[] Mnode_max_key_;
    }
  }

  AlexModelNode(const self_type& other)
      : AlexNode<P>(other),
        allocator_(other.allocator_),
        num_children_(other.num_children_) {
    children_ = new (pointer_allocator().allocate(other.num_children_))
        AlexNode<P>*[other.num_children_];
    std::copy(other.children_, other.children_ + other.num_children_,
              children_);

    if (other.Mnode_min_key != nullptr) {
      Mnode_min_key_ = new double[this->max_key_length_];
      std::copy(other.Mnode_min_key_, other.Mnode_min_key_ + this->max_key_length_, Mnode_min_key_);
    }
    if (other.Mnode_max_key != nullptr) {
      Mnode_max_key_ = new double[this->max_key_length_];
      std::copy(other.Mnode_max_key_, other.Mnode_max_key_ + this->max_key_length_, Mnode_max_key_);
    }
  }

  // Given a key, traverses to the child node responsible for that key
  inline AlexNode<P>* get_child_node(const AlexKey& key) {
    int bucketID = this->model_.predict(key);
    bucketID = std::min<int>(std::max<int>(bucketID, 0), num_children_ - 1);
    return children_[bucketID];
  }

  // Expand by a power of 2 by creating duplicates of all existing child
  // pointers.
  // Input is the base 2 log of the expansion factor, in order to guarantee
  // expanding by a power of 2.
  // Returns the expansion factor.
  int expand(int log2_expansion_factor) {
    assert(log2_expansion_factor >= 0);
    int expansion_factor = 1 << log2_expansion_factor;
    int num_new_children = num_children_ * expansion_factor;
    auto new_children = new (pointer_allocator().allocate(num_new_children))
        AlexNode<P>*[num_new_children];
    int cur = 0;
    while (cur < num_children_) {
      AlexNode<P>* cur_child = children_[cur];
      int cur_child_repeats = 1 << cur_child->duplication_factor_;
      for (int i = expansion_factor * cur;
           i < expansion_factor * (cur + cur_child_repeats); i++) {
        new_children[i] = cur_child;
      }
      cur_child->duplication_factor_ += log2_expansion_factor;
      cur += cur_child_repeats;
    }
    pointer_allocator().deallocate(children_, num_children_);
    children_ = new_children;
    num_children_ = num_new_children;
    this->model_.expand(expansion_factor);
    return expansion_factor;
  }

  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  long long node_size() const override {
    long long size = sizeof(self_type);
    size += num_children_ * sizeof(AlexNode<P>*);  // pointers to children
    return size;
  }

  // Helpful for debugging
  bool validate_structure(bool verbose = false) const {
    if (num_children_ == 0) {
      if (verbose) {
        std::cout << "[Childless node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }
    if (num_children_ == 1) {
      if (verbose) {
        std::cout << "[Single child node] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }
    if (std::ceil(std::log2(num_children_)) !=
        std::floor(std::log2(num_children_))) {
      if (verbose) {
        std::cout << "[Num children not a power of 2] num children: "
                  << num_children_ << std::endl;
      }
      return false;
    }

    int zero_slope = 1;
    for (int i = 0; i < this->model_.a_.max_key_length_; i++) {
      if (this->model_.a_[i] != 0.0) {zero_slope = 0; break;}
    }
    if (zero_slope) {
      if (verbose) {
        std::cout << "[Model node with zero slope] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    AlexNode<P>* cur_child = children_[0];
    int cur_repeats = 1;
    int i;
    for (i = 1; i < num_children_; i++) {
      if (children_[i] == cur_child) {
        cur_repeats++;
      } else {
        if (cur_repeats != (1 << cur_child->duplication_factor_)) {
          if (verbose) {
            std::cout << "[Incorrect duplication factor] num actual repeats: "
                      << cur_repeats << ", num dup_factor repeats: "
                      << (1 << cur_child->duplication_factor_)
                      << ", parent addr: " << this
                      << ", parent level: " << this->level_
                      << ", parent num children: " << num_children_
                      << ", child addr: " << children_[i - cur_repeats]
                      << ", child pointer indexes: [" << i - cur_repeats << ", "
                      << i << ")" << std::endl;
          }
          return false;
        }
        if (std::ceil(std::log2(cur_repeats)) !=
            std::floor(std::log2(cur_repeats))) {
          if (verbose) {
            std::cout
                << "[Num duplicates not a power of 2] num actual repeats: "
                << cur_repeats << std::endl;
          }
          return false;
        }
        if (i % cur_repeats != 0) {
          if (verbose) {
            std::cout
                << "[Duplicate region incorrectly aligned] num actual repeats: "
                << cur_repeats << ", num dup_factor repeats: "
                << (1 << cur_child->duplication_factor_)
                << ", child pointer indexes: [" << i - cur_repeats << ", " << i
                << ")" << std::endl;
          }
          return false;
        }
        cur_child = children_[i];
        cur_repeats = 1;
      }
    }
    if (cur_repeats != (1 << cur_child->duplication_factor_)) {
      if (verbose) {
        std::cout << "[Incorrect duplication factor] num actual repeats: "
                  << cur_repeats << ", num dup_factor repeats: "
                  << (1 << cur_child->duplication_factor_)
                  << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_
                  << ", child addr: " << children_[i - cur_repeats]
                  << ", child pointer indexes: [" << i - cur_repeats << ", "
                  << i << ")" << std::endl;
      }
      return false;
    }
    if (std::ceil(std::log2(cur_repeats)) !=
        std::floor(std::log2(cur_repeats))) {
      if (verbose) {
        std::cout << "[Num duplicates not a power of 2] num actual repeats: "
                  << cur_repeats << std::endl;
      }
      return false;
    }
    if (i % cur_repeats != 0) {
      if (verbose) {
        std::cout
            << "[Duplicate region incorrectly aligned] num actual repeats: "
            << cur_repeats << ", num dup_factor repeats: "
            << (1 << cur_child->duplication_factor_)
            << ", child pointer indexes: [" << i - cur_repeats << ", " << i
            << ")" << std::endl;
      }
      return false;
    }
    if (cur_repeats == num_children_) {
      if (verbose) {
        std::cout << "[All children are the same] num actual repeats: "
                  << cur_repeats << ", parent addr: " << this
                  << ", parent level: " << this->level_
                  << ", parent num children: " << num_children_ << std::endl;
      }
      return false;
    }

    return true;
  }
};

/*
* Functions are organized into different sections:
* - Constructors and destructors
* - General helper functions
* - Iterator
* - Cost model
* - Bulk loading and model building (e.g., bulk_load, bulk_load_from_existing)
* - Lookups (e.g., find_key, find_lower, find_upper, lower_bound, upper_bound)
* - Inserts and resizes (e.g, insert)
* - Deletes (e.g., erase, erase_one)
* - Stats
* - Debugging
*/
template <class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<AlexKey, P>>,
          bool allow_duplicates = true>
class AlexDataNode : public AlexNode<P> {
 public:
  typedef std::pair<AlexKey, P> V;
  typedef AlexNode<P> basic_node_type;
  typedef AlexModelNode<P, Alloc> model_node_type;
  typedef AlexDataNode<P, Compare, Alloc, allow_duplicates> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<AlexKey>::other key_alloc_type;
  typedef typename Alloc::template rebind<P>::other payload_alloc_type;
  typedef typename Alloc::template rebind<V>::other value_alloc_type;
  typedef typename Alloc::template rebind<uint64_t>::other bitmap_alloc_type;

  const Compare& key_less_;
  const Alloc& allocator_;

  // Forward declaration
  template <typename node_type = self_type, typename payload_return_type = P,
            typename value_return_type = V>
  class Iterator;
  typedef Iterator<> iterator_type;
  typedef Iterator<const self_type, const P, const V> const_iterator_type;

  self_type* next_leaf_ = nullptr;
  self_type* prev_leaf_ = nullptr;

#if ALEX_DATA_NODE_SEP_ARRAYS
  AlexKey* key_slots_ = nullptr;  // holds keys
  P* payload_slots_ =
      nullptr;  // holds payloads, must be same size as key_slots
#else
  V* data_slots_ = nullptr;  // holds key-payload pairs
#endif

  /* Below are unused attributes */
  //unsigned int max_key_length_ = 1; // maximum length of each key 
  //int key_type_ = DOUBLE; // key type for specific node.

  int data_capacity_ = 0;  // size of key/data_slots array
  int num_keys_ = 0;  // number of filled key/data slots (as opposed to gaps)
  double *the_max_key_arr_; //theoretic maximum key_arr
  double *the_min_key_arr_; //theoretic minimum key_arr

  // Bitmap: each uint64_t represents 64 positions in reverse order
  // (i.e., each uint64_t is "read" from the right-most bit to the left-most
  // bit)
  uint64_t* bitmap_ = nullptr;
  int bitmap_size_ = 0;  // number of int64_t in bitmap

  // Variables related to resizing (expansions and contractions)
  static constexpr double kMaxDensity_ = 0.8;  // density after contracting,
                                               // also determines the expansion
                                               // threshold
  static constexpr double kInitDensity_ =
      0.7;  // density of data nodes after bulk loading
  static constexpr double kMinDensity_ = 0.6;  // density after expanding, also
                                               // determines the contraction
                                               // threshold
  double expansion_threshold_ = 1;  // expand after m_num_keys is >= this number
  double contraction_threshold_ =
      0;  // contract after m_num_keys is < this number
  static constexpr int kDefaultMaxDataNodeBytes_ =
      1 << 24;  // by default, maximum data node size is 16MB
  int max_slots_ =
      kDefaultMaxDataNodeBytes_ /
      sizeof(V);  // cannot expand beyond this number of key/data slots

  // Counters used in cost models
  long long num_shifts_ = 0;                 // does not reset after resizing
  long long num_exp_search_iterations_ = 0;  // does not reset after resizing
  int num_lookups_ = 0;                      // does not reset after resizing
  int num_inserts_ = 0;                      // does not reset after resizing
  int num_resizes_ = 0;  // technically not required, but nice to have

  // Variables for determining append-mostly behavior
  // max key in node, updates after inserts but not erases.
  AlexKey *max_key_; 
  // min key in node, updates after inserts but not erases. 
  AlexKey *min_key_;
  // mid key in node, updates after inserts but not erases.
  AlexKey *mid_key_ = new AlexKey(0); 
  int num_right_out_of_bounds_inserts_ =
      0;  // number of inserts that are larger than the max key
  int num_left_out_of_bounds_inserts_ =
      0;  // number of inserts that are smaller than the min key
  // Node is considered append-mostly if the fraction of inserts that are out of
  // bounds is above this threshold
  // Append-mostly nodes will expand in a manner that anticipates further
  // appends
  static constexpr double kAppendMostlyThreshold = 0.9;

  // Purely for benchmark debugging purposes
  double expected_avg_exp_search_iterations_ = 0;
  double expected_avg_shifts_ = 0;

  // Placed at the end of the key/data slots if there are gaps after the max key.
  // It was originally static constexpr, but I changed to normal AlexKey.
  AlexKey kEndSentinel_; 

  /*** Constructors and destructors ***/

  AlexDataNode () : AlexNode<P>(0, true) {
    the_max_key_arr_ = new double[1];
    the_min_key_arr_ = new double[1];
    the_max_key_arr_[0] = std::numeric_limits<double>::max();
    the_min_key_arr_[0] = std::numeric_limits<double>::min();

    double *max_key = new double[1];
    double *min_key = new double[1];
    double *kEndSentinel = new double[1];
    max_key[0] = std::numeric_limits<double>::max();
    min_key[0] = std::numeric_limits<double>::lowest();
    kEndSentinel[0] = std::numeric_limits<double>::max();
    max_key_ = new AlexKey(max_key, 1);
    min_key_ = new AlexKey(min_key, 1);
    kEndSentinel_.key_arr_ = kEndSentinel;
    kEndSentinel_.max_key_length_ = 1;
  }

  explicit AlexDataNode(unsigned int max_key_length, int key_type, model_node_type *parent,
        const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<P>(0, true, parent, max_key_length), key_less_(comp), allocator_(alloc) {
    double *max_key_arr = new double[this->max_key_length_];
    double *min_key_arr = new double[this->max_key_length_];
    double *mid_key_arr = new double[this->max_key_length_]();
    double *kEndSentinel_arr = new double[this->max_key_length_];
    the_max_key_arr_ = new double[this->max_key_length_];
    the_min_key_arr_ = new double[this->max_key_length_];
    min_key_ = new AlexKey(min_key_arr, this->max_key_length_);
    max_key_ = new AlexKey(max_key_arr, this->max_key_length_);
    mid_key_ = new AlexKey(mid_key_arr, this->max_key_length_);
    kEndSentinel_.key_arr_ = kEndSentinel_arr;
    kEndSentinel_.max_key_length_ = this->max_key_length_;

    switch (key_type) {
      case DOUBLE:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_,
            std::numeric_limits<double>::lowest());
        std::fill(min_key_arr, min_key_arr + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_,
            std::numeric_limits<double>::lowest());
        break;
      case INTEGER:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_,
            std::numeric_limits<int>::lowest());
        std::fill(min_key_arr, min_key_arr + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_,
            std::numeric_limits<int>::lowest());
        break;
      case STRING:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_, 0.0);
        std::fill(min_key_arr, min_key_arr + this->max_key_length_, 127.0);
        std::fill(mid_key_arr, mid_key_arr + this->max_key_length_, 63.0);
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_, 127.0);
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_, 127.0);
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_, 0.0);
        break;  
    }
  }

  AlexDataNode(short level, int max_data_node_slots,
               unsigned int max_key_length, int key_type, basic_node_type *parent,
               const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<P>(level, true, parent, max_key_length),
        key_less_(comp),
        allocator_(alloc),
        max_slots_(max_data_node_slots) {
    double *max_key_arr = new double[this->max_key_length_];
    double *min_key_arr = new double[this->max_key_length_];
    double *mid_key_arr = new double[this->max_key_length_]();
    double *kEndSentinel_arr = new double[this->max_key_length_];
    the_max_key_arr_ = new double[this->max_key_length_];
    the_min_key_arr_ = new double[this->max_key_length_];
    min_key_ = new AlexKey(min_key_arr, this->max_key_length_);
    max_key_ = new AlexKey(max_key_arr, this->max_key_length_);
    mid_key_ = new AlexKey(mid_key_arr, this->max_key_length_);
    kEndSentinel_.key_arr_ = kEndSentinel_arr;
    kEndSentinel_.max_key_length_ = this->max_key_length_;

    switch (key_type) {
      case DOUBLE:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_,
            std::numeric_limits<double>::lowest());
        std::fill(min_key_arr, min_key_arr + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_,
            std::numeric_limits<double>::max());
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_,
            std::numeric_limits<double>::lowest());
        break;
      case INTEGER:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_,
            std::numeric_limits<int>::lowest());
        std::fill(min_key_arr, min_key_arr + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_,
            std::numeric_limits<int>::max());
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_,
            std::numeric_limits<int>::lowest());
        break;
      case STRING:
        std::fill(max_key_arr, max_key_arr + this->max_key_length_, 0.0);
        std::fill(min_key_arr, min_key_arr + this->max_key_length_, 127.0);
        std::fill(mid_key_arr, mid_key_arr + this->max_key_length_, 63.0);
        std::fill(kEndSentinel_arr, kEndSentinel_arr + this->max_key_length_, 127.0);
        std::fill(the_max_key_arr_, the_max_key_arr_ + this->max_key_length_, 127.0);
        std::fill(the_min_key_arr_, the_min_key_arr_ + this->max_key_length_, 0.0);
        break;  
    }
  }

  ~AlexDataNode() {
#if ALEX_DATA_NODE_SEP_ARRAYS
    if (key_slots_ == nullptr) {
      return;
    }
    key_allocator().deallocate(key_slots_, data_capacity_);
    payload_allocator().deallocate(payload_slots_, data_capacity_);
#else
    if (data_slots_ == nullptr) {
      return;
    }
    value_allocator().deallocate(data_slots_, data_capacity_);
#endif
    bitmap_allocator().deallocate(bitmap_, bitmap_size_);
    delete min_key_;
    delete max_key_;
    delete mid_key_;
    delete[] kEndSentinel_.key_arr_;
    delete[] the_max_key_arr_;
    delete[] the_min_key_arr_;
  }

  AlexDataNode(const self_type& other)
      : AlexNode<P>(other),
        key_less_(other.key_less_),
        allocator_(other.allocator_),
        next_leaf_(other.next_leaf_),
        prev_leaf_(other.prev_leaf_),
        data_capacity_(other.data_capacity_),
        num_keys_(other.num_keys_),
        bitmap_size_(other.bitmap_size_),
        expansion_threshold_(other.expansion_threshold_),
        contraction_threshold_(other.contraction_threshold_),
        max_slots_(other.max_slots_),
        num_shifts_(other.num_shifts_),
        num_exp_search_iterations_(other.num_exp_search_iterations_),
        num_lookups_(other.num_lookups_),
        num_inserts_(other.num_inserts_),
        num_resizes_(other.num_resizes_),
        num_right_out_of_bounds_inserts_(
            other.num_right_out_of_bounds_inserts_),
        num_left_out_of_bounds_inserts_(other.num_left_out_of_bounds_inserts_),
        expected_avg_exp_search_iterations_(
            other.expected_avg_exp_search_iterations_),
        expected_avg_shifts_(other.expected_avg_shifts_) {
    /* deep copy of max/min_key array is needed
     * since deletion of either one of the datanode
     * would result to other one's key's data pointer invalid
     * for similar reason, kEndSentinel_ also needs deep copying. */
    double *max_key_arr = new double[this->max_key_length_];
    double *mid_key_arr = new double[this->max_key_length_];
    double *min_key_arr = new double[this->max_key_length_];
    double *kEndSentinel_arr = new double[this->max_key_length_];
    double *the_max_key_arr = new double[this->max_key_length_];
    double *the_min_key_arr = new double[this->max_key_length_];
    std::copy(other.max_key_->key_arr_, other.max_key_->key_arr_ + this->max_key_length_,
        max_key_arr);
    std::copy(other.min_key_->key_arr_, other.min_key_->key_arr_ + this->max_key_length_,
        min_key_arr);
    std::copy(other.mid_key_->key_arr_, other.mid_key_->key_arr_ + this->max_key_length_,
        mid_key_arr);
    std::copy(other.kEndSentinel_.key_arr_, other.kEndSentinel_.key_arr_ + this->max_key_length_,
        kEndSentinel_arr);
    std::copy(other.the_max_key_arr_, other.the_max_key_arr_ + this->max_key_length_,
        the_max_key_arr);
    std::copy(other.the_min_key_arr_, other.the_min_key_arr_ + this->max_key_length_,
        the_min_key_arr);
    max_key_ = new AlexKey(max_key_arr, this->max_key_length_);
    min_key_ = new AlexKey(min_key_arr, this->max_key_length_);
    mid_key_ = new AlexKey(mid_key_arr, this->max_key_length_);
    kEndSentinel_.key_arr_ = kEndSentinel_arr;
    kEndSentinel_.max_key_length_ = this->max_key_length_;
    the_max_key_arr_ = the_max_key_arr;
    the_min_key_arr_ = the_min_key_arr;

#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new (key_allocator().allocate(other.data_capacity_))
        AlexKey[other.data_capacity_];
    std::copy(other.key_slots_, other.key_slots_ + other.data_capacity_,
              key_slots_);
    payload_slots_ = new (payload_allocator().allocate(other.data_capacity_))
        P[other.data_capacity_];
    std::copy(other.payload_slots_, other.payload_slots_ + other.data_capacity_,
              payload_slots_);
#else
    data_slots_ = new (value_allocator().allocate(other.data_capacity_))
        V[other.data_capacity_];
    std::copy(other.data_slots_, other.data_slots_ + other.data_capacity_,
              data_slots_);
#endif
    bitmap_ = new (bitmap_allocator().allocate(other.bitmap_size_))
        uint64_t[other.bitmap_size_];
    std::copy(other.bitmap_, other.bitmap_ + other.bitmap_size_, bitmap_);
  }

  /*** Allocators ***/

  key_alloc_type key_allocator() { return key_alloc_type(allocator_); }

  payload_alloc_type payload_allocator() {
    return payload_alloc_type(allocator_);
  }

  value_alloc_type value_allocator() { return value_alloc_type(allocator_); }

  bitmap_alloc_type bitmap_allocator() { return bitmap_alloc_type(allocator_); }

  /*** General helper functions ***/

  inline AlexKey& get_key(int pos) const { return ALEX_DATA_NODE_KEY_AT(pos); }

  //newly added for actual content achieving without need for max_length data.
  inline double *get_key_arr(int pos) const { return get_key(pos).key_arr_; }

  inline P& get_payload(int pos) const {
    return ALEX_DATA_NODE_PAYLOAD_AT(pos);
  }

  // Check whether the position corresponds to a key (as opposed to a gap)
  inline bool check_exists(int pos) const {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    return static_cast<bool>(bitmap_[bitmap_pos] & (1ULL << bit_pos));
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Mark the entry for position in the bitmap
  inline void set_bit(uint64_t bitmap[], int pos) {
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap[bitmap_pos] |= (1ULL << bit_pos);
  }

  // Unmark the entry for position in the bitmap
  inline void unset_bit(int pos) {
    assert(pos >= 0 && pos < data_capacity_);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    bitmap_[bitmap_pos] &= ~(1ULL << bit_pos);
  }

  // Value of first (i.e., min) key
  double * first_key() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return get_key_arr(i);
    }
    return the_max_key_arr_;
  }

  // Value of last (i.e., max) key
  double * last_key() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return get_key_arr(i);
    }
    return the_min_key_arr_;
  }

  // Position in key/data_slots of first (i.e., min) key
  int first_pos() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Position in key/data_slots of last (i.e., max) key
  int last_pos() const {
    for (int i = data_capacity_ - 1; i >= 0; i--) {
      if (check_exists(i)) return i;
    }
    return 0;
  }

  // Number of keys between positions left and right (exclusive) in
  // key/data_slots
  int num_keys_in_range(int left, int right) const {
    assert(left >= 0 && left <= right && right <= data_capacity_);
    int num_keys = 0;
    int left_bitmap_idx = left >> 6;
    int right_bitmap_idx = right >> 6;
    if (left_bitmap_idx == right_bitmap_idx) {
      uint64_t bitmap_data = bitmap_[left_bitmap_idx];
      int left_bit_pos = left - (left_bitmap_idx << 6);
      bitmap_data &= ~((1ULL << left_bit_pos) - 1);
      int right_bit_pos = right - (right_bitmap_idx << 6);
      bitmap_data &= ((1ULL << right_bit_pos) - 1);
      num_keys += _mm_popcnt_u64(bitmap_data);
    } else {
      uint64_t left_bitmap_data = bitmap_[left_bitmap_idx];
      int bit_pos = left - (left_bitmap_idx << 6);
      left_bitmap_data &= ~((1ULL << bit_pos) - 1);
      num_keys += _mm_popcnt_u64(left_bitmap_data);
      for (int i = left_bitmap_idx + 1; i < right_bitmap_idx; i++) {
        num_keys += _mm_popcnt_u64(bitmap_[i]);
      }
      if (right_bitmap_idx != bitmap_size_) {
        uint64_t right_bitmap_data = bitmap_[right_bitmap_idx];
        bit_pos = right - (right_bitmap_idx << 6);
        right_bitmap_data &= ((1ULL << bit_pos) - 1);
        num_keys += _mm_popcnt_u64(right_bitmap_data);
      }
    }
    return num_keys;
  }

  // True if a < b
  //template <class K>
  forceinline bool key_less(const AlexKey& a, const AlexKey& b) const {
    return key_less_(a, b);
  }

  // True if a <= b
  //template <class K>
  forceinline bool key_lessequal(const AlexKey& a, const AlexKey& b) const {
    return !key_less_(b, a);
  }

  // True if a > b
  //template <class K>
  forceinline bool key_greater(const AlexKey& a, const AlexKey& b) const {
    return key_less_(b, a);
  }

  // True if a >= b
  //template <class K>
  forceinline bool key_greaterequal(const AlexKey& a, const AlexKey& b) const {
    return !key_less_(a, b);
  }

  // True if a == b
  //template <class K>
  forceinline bool key_equal(const AlexKey& a, const AlexKey& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Iterator ***/

  // Forward iterator meant for iterating over a single data node.
  // By default, it is a "normal" non-const iterator.
  // Can be templated to be a const iterator.
  template <typename node_type, typename payload_return_type,
            typename value_return_type>
  class Iterator {
   public:
    node_type* node_;
    int cur_idx_ = 0;  // current position in key/data_slots, -1 if at end
    int cur_bitmap_idx_ = 0;  // current position in bitmap
    uint64_t cur_bitmap_data_ =
        0;  // caches the relevant data in the current bitmap position

    explicit Iterator(node_type* node) : node_(node) {}

    Iterator(node_type* node, int idx) : node_(node), cur_idx_(idx) {
      initialize();
    }

    void initialize() {
      cur_bitmap_idx_ = cur_idx_ >> 6;
      cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];

      // Zero out extra bits
      int bit_pos = cur_idx_ - (cur_bitmap_idx_ << 6);
      cur_bitmap_data_ &= ~((1ULL << bit_pos) - 1);

      (*this)++;
    }

    void operator++(int) {
      while (cur_bitmap_data_ == 0) {
        cur_bitmap_idx_++;
        if (cur_bitmap_idx_ >= node_->bitmap_size_) {
          cur_idx_ = -1;
          return;
        }
        cur_bitmap_data_ = node_->bitmap_[cur_bitmap_idx_];
      }
      uint64_t bit = extract_rightmost_one(cur_bitmap_data_);
      cur_idx_ = get_offset(cur_bitmap_idx_, bit);
      cur_bitmap_data_ = remove_rightmost_one(cur_bitmap_data_);
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    V operator*() const {
      return std::make_pair(node_->key_slots_[cur_idx_],
                            node_->payload_slots_[cur_idx_]);
    }
#else
    value_return_type& operator*() const {
      return node_->data_slots_[cur_idx_];
    }
#endif

    const AlexKey& key() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return node_->key_slots_[cur_idx_];
#else
      return node_->data_slots_[cur_idx_].first;
#endif
    }

    payload_return_type& payload() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return node_->payload_slots_[cur_idx_];
#else
      return node_->data_slots_[cur_idx_].second;
#endif
    }

    bool is_end() const { return cur_idx_ == -1; }

    bool operator==(const Iterator& rhs) const {
      return cur_idx_ == rhs.cur_idx_;
    }

    bool operator!=(const Iterator& rhs) const { return !(*this == rhs); };
  };

  iterator_type begin() { return iterator_type(this, 0); }

  /*** Cost model ***/

  // Empirical average number of shifts per insert
  double shifts_per_insert() const {
    if (num_inserts_ == 0) {
      return 0;
    }
    return num_shifts_ / static_cast<double>(num_inserts_);
  }

  // Empirical average number of exponential search iterations per operation
  // (either lookup or insert)
  double exp_search_iterations_per_operation() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    return num_exp_search_iterations_ /
           static_cast<double>(num_inserts_ + num_lookups_);
  }

  double empirical_cost() const {
    if (num_inserts_ + num_lookups_ == 0) {
      return 0;
    }
    double frac_inserts =
        static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
    return kExpSearchIterationsWeight * exp_search_iterations_per_operation() +
           kShiftsWeight * shifts_per_insert() * frac_inserts;
  }

  // Empirical fraction of operations (either lookup or insert) that are inserts
  double frac_inserts() const {
    int num_ops = num_inserts_ + num_lookups_;
    if (num_ops == 0) {
      return 0;  // if no operations, assume no inserts
    }
    return static_cast<double>(num_inserts_) / (num_inserts_ + num_lookups_);
  }

  void reset_stats() {
    num_shifts_ = 0;
    num_exp_search_iterations_ = 0;
    num_lookups_ = 0;
    num_inserts_ = 0;
    num_resizes_ = 0;
  }

  // Computes the expected cost of the current node
  double compute_expected_cost(double frac_inserts = 0) {
    if (num_keys_ == 0) {
      return 0;
    }

    ExpectedSearchIterationsAccumulator search_iters_accumulator;
    ExpectedShiftsAccumulator shifts_accumulator(data_capacity_);
    const_iterator_type it(this, 0);
    for (; !it.is_end(); it++) {
      int predicted_position = std::max(
          0, std::min(data_capacity_ - 1, this->model_.predict(it.key())));
      search_iters_accumulator.accumulate(it.cur_idx_, predicted_position);
      shifts_accumulator.accumulate(it.cur_idx_, predicted_position);
    }
    expected_avg_exp_search_iterations_ = search_iters_accumulator.get_stat();
    expected_avg_shifts_ = shifts_accumulator.get_stat();
    double cost =
        kExpSearchIterationsWeight * expected_avg_exp_search_iterations_ +
        kShiftsWeight * expected_avg_shifts_ * frac_inserts;
    return cost;
  }

  // Computes the expected cost of a data node constructed using the input dense
  // array of keys
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost(
      const V* values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel* existing_model = nullptr, bool use_sampling = false,
      DataNodeStats* stats = nullptr) {
    if (use_sampling) {
      return compute_expected_cost_sampling(values, num_keys, density,
                                            expected_insert_frac,
                                            existing_model, stats);
    }

    if (num_keys == 0) {
      return 0;
    }

    int data_capacity =
        std::max(static_cast<int>(num_keys / density), num_keys + 1);

    // Compute what the node's model would be
    LinearModel model;
    if (existing_model == nullptr) {
      model.max_key_length_ = values[0].first.max_key_length_;
      build_model(values, num_keys, &model);
    } else {
      model.max_key_length_ = existing_model->max_key_length_;
      model.a_ = new double[model.max_key_length_];
      for (unsigned int i = 0; i < model.max_key_length_; i++) {
        model.a_[i] = existing_model->a_[i];
      }
      model.b_ = existing_model->b_;
    }
    model.expand(static_cast<double>(data_capacity) / num_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;
    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit(values, num_keys, data_capacity, &acc, &model);
      expected_avg_exp_search_iterations =
          acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }
    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
           kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit(const V* values, int num_keys,
                                  int data_capacity, StatAccumulator* acc,
                                  const LinearModel* model) {
    int last_position = -1;
    int keys_remaining = num_keys;
    for (int i = 0; i < num_keys; i++) {
      int predicted_position = std::max(
          0, std::min(data_capacity - 1, model->predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;
      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (int j = i; j < num_keys; j++) {
          predicted_position = std::max(
              0, std::min(data_capacity - 1, model->predict(values[j].first)));
          acc->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }
      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  // Using sampling, approximates the expected cost of a data node constructed
  // using the input dense array of keys
  // Assumes existing_model is trained on the dense array of keys
  // Uses progressive sampling: keep increasing the sample size until the
  // computed stats stop changing drastically
  static double compute_expected_cost_sampling(
      const V* values, int num_keys, double density,
      double expected_insert_frac,
      const LinearModel* existing_model = nullptr,
      DataNodeStats* stats = nullptr) {
    const static int min_sample_size = 25;

    // Stop increasing sample size if relative diff of stats between samples is
    // less than this
    const static double rel_diff_threshold = 0.2;

    // Equivalent threshold in log2-space
    const static double abs_log2_diff_threshold =
        std::log2(1 + rel_diff_threshold);

    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If num_keys is below this threshold, we compute entropy exactly
    const static int exact_computation_size_threshold =
        (min_sample_size * sample_size_multiplier * sample_size_multiplier * 2);

    // Target fraction of the keys to use in the initial sample
    const static double init_sample_frac = 0.01;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys < exact_computation_size_threshold) {
      return compute_expected_cost(values, num_keys, density,
                                   expected_insert_frac, existing_model, false,
                                   stats);
    }

    LinearModel model;  // trained for full dense array
    if (existing_model == nullptr) {
      model.max_key_length_ = values[0].first.max_key_length_;
      build_model(values, num_keys, &model);
    } else {
      model.max_key_length_ = existing_model->max_key_length_;
      model.a_ = new double[model.max_key_length_];
      for (int i = 0; i < model.max_key_length_; i++) {
        model.a_[i] = existing_model->a_[i];
      }
      model.b_ = existing_model->b_;
    }

    // Compute initial sample size and step size
    // Right now, sample_num_keys holds the target sample num keys
    int sample_num_keys = std::max(
        static_cast<int>(num_keys * init_sample_frac), min_sample_size);
    int step_size = 1;
    double tmp_sample_size =
        num_keys;  // this helps us determine the right sample size
    while (tmp_sample_size >= sample_num_keys) {
      tmp_sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;
    sample_num_keys =
        num_keys /
        step_size;  // now sample_num_keys is the actual sample num keys

    std::vector<SampleDataNodeStats>
        sample_stats;  // stats computed usinig each sample
    bool compute_shifts = expected_insert_frac !=
                          0;  // whether we need to compute expected shifts
    double log2_num_keys = std::log2(num_keys);
    double expected_full_search_iters =
        0;  // extrapolated estimate for search iters on the full array
    double expected_full_shifts =
        0;  // extrapolated estimate shifts on the full array
    bool search_iters_computed =
        false;  // set to true when search iters is accurately computed
    bool shifts_computed =
        false;  // set to true when shifts is accurately computed

    // Progressively increase sample size
    while (true) {
      int sample_data_capacity = std::max(
          static_cast<int>(sample_num_keys / density), sample_num_keys + 1);
      LinearModel sample_model(model.a_, model.b_, model.max_key_length_);
      sample_model.expand(static_cast<double>(sample_data_capacity) / num_keys);

      // Compute stats using the sample
      if (expected_insert_frac == 0) {
        ExpectedSearchIterationsAccumulator acc;
        build_node_implicit_sampling(values, num_keys, sample_num_keys,
                                     sample_data_capacity, step_size, &acc,
                                     &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys), acc.get_stat(), 0});
      } else {
        ExpectedIterationsAndShiftsAccumulator acc(sample_data_capacity);
        build_node_implicit_sampling(values, num_keys, sample_num_keys,
                                     sample_data_capacity, step_size, &acc,
                                     &sample_model);
        sample_stats.push_back({std::log2(sample_num_keys),
                                acc.get_expected_num_search_iterations(),
                                std::log2(acc.get_expected_num_shifts())});
      }

      if (sample_stats.size() >= 3) {
        // Check if the diff in stats is sufficiently small
        SampleDataNodeStats& s0 = sample_stats[sample_stats.size() - 3];
        SampleDataNodeStats& s1 = sample_stats[sample_stats.size() - 2];
        SampleDataNodeStats& s2 = sample_stats[sample_stats.size() - 1];
        // (y1 - y0) / (x1 - x0) = (y2 - y1) / (x2 - x1) --> y2 = (y1 - y0) /
        // (x1 - x0) * (x2 - x1) + y1
        double expected_s2_search_iters =
            (s1.num_search_iterations - s0.num_search_iterations) /
                (s1.log2_sample_size - s0.log2_sample_size) *
                (s2.log2_sample_size - s1.log2_sample_size) +
            s1.num_search_iterations;
        double rel_diff =
            std::abs((s2.num_search_iterations - expected_s2_search_iters) /
                     s2.num_search_iterations);
        if (rel_diff <= rel_diff_threshold || num_keys <= 2 * sample_num_keys) {
          search_iters_computed = true;
          expected_full_search_iters =
              (s2.num_search_iterations - s1.num_search_iterations) /
                  (s2.log2_sample_size - s1.log2_sample_size) *
                  (log2_num_keys - s2.log2_sample_size) +
              s2.num_search_iterations;
        }
        if (compute_shifts) {
          double expected_s2_log2_shifts =
              (s1.log2_num_shifts - s0.log2_num_shifts) /
                  (s1.log2_sample_size - s0.log2_sample_size) *
                  (s2.log2_sample_size - s1.log2_sample_size) +
              s1.log2_num_shifts;
          double abs_diff =
              std::abs((s2.log2_num_shifts - expected_s2_log2_shifts) /
                       s2.log2_num_shifts);
          if (abs_diff <= abs_log2_diff_threshold ||
              num_keys <= 2 * sample_num_keys) {
            shifts_computed = true;
            double expected_full_log2_shifts =
                (s2.log2_num_shifts - s1.log2_num_shifts) /
                    (s2.log2_sample_size - s1.log2_sample_size) *
                    (log2_num_keys - s2.log2_sample_size) +
                s2.log2_num_shifts;
            expected_full_shifts = std::pow(2, expected_full_log2_shifts);
          }
        }

        // If diff in stats is sufficiently small, return the approximate
        // expected cost
        if ((expected_insert_frac == 0 && search_iters_computed) ||
            (expected_insert_frac > 0 && search_iters_computed &&
             shifts_computed)) {
          double cost =
              kExpSearchIterationsWeight * expected_full_search_iters +
              kShiftsWeight * expected_full_shifts * expected_insert_frac;
          if (stats) {
            stats->num_search_iterations = expected_full_search_iters;
            stats->num_shifts = expected_full_shifts;
          }
          return cost;
        }
      }

      step_size /= sample_size_multiplier;
      sample_num_keys = num_keys / step_size;
    }
  }

  // Helper function for compute_expected_cost_sampling
  // Implicitly build the data node in order to collect the stats
  // keys is the full un-sampled array of keys
  // sample_num_keys and sample_data_capacity refer to a data node that is
  // created only over the sample
  // sample_model is trained for the sampled data node
  static void build_node_implicit_sampling(const V* values, int num_keys,
                                           int sample_num_keys,
                                           int sample_data_capacity,
                                           int step_size, StatAccumulator* ent,
                                           const LinearModel* sample_model) {
    int last_position = -1;
    int sample_keys_remaining = sample_num_keys;
    for (int i = 0; i < num_keys; i += step_size) {
      int predicted_position =
          std::max(0, std::min(sample_data_capacity - 1,
                               sample_model->predict(values[i].first)));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = sample_data_capacity - actual_position;
      if (positions_remaining < sample_keys_remaining) {
        actual_position = sample_data_capacity - sample_keys_remaining;
        for (int j = i; j < num_keys; j += step_size) {
          predicted_position =
              std::max(0, std::min(sample_data_capacity - 1,
                                   sample_model->predict(values[j].first)));
          ent->accumulate(actual_position, predicted_position);
          actual_position++;
        }
        break;
      }
      ent->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      sample_keys_remaining--;
    }
  }

  // Computes the expected cost of a data node constructed using the keys
  // between left and right in the
  // key/data_slots of an existing node
  // Assumes existing_model is trained on the dense array of keys
  static double compute_expected_cost_from_existing(
      const self_type* node, int left, int right, double density,
      double expected_insert_frac,
      const LinearModel* existing_model = nullptr,
      DataNodeStats* stats = nullptr) {
    assert(left >= 0 && right <= node->data_capacity_);

    LinearModel model;
    int num_actual_keys = 0;
    if (existing_model == nullptr) {
      model.max_key_length_ = node->max_key_length_;
      const_iterator_type it(node, left);
      LinearModelBuilder builder(&model);
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = node->num_keys_in_range(left, right);
      model.max_key_length_ = existing_model->max_key_length_;
      model.a_ = new double[model.max_key_length_];
      for (int i = 0; i < model.max_key_length_; i++) {
        model.a_[i] = existing_model->a_[i];
      }
      model.b_ = existing_model->b_;
    }

    if (num_actual_keys == 0) {
      return 0;
    }
    int data_capacity = std::max(static_cast<int>(num_actual_keys / density),
                                 num_actual_keys + 1);
    model.expand(static_cast<double>(data_capacity) / num_actual_keys);

    // Compute expected stats in order to compute the expected cost
    double cost = 0;
    double expected_avg_exp_search_iterations = 0;
    double expected_avg_shifts = 0;
    if (expected_insert_frac == 0) {
      ExpectedSearchIterationsAccumulator acc;
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);
      expected_avg_exp_search_iterations = acc.get_stat();
    } else {
      ExpectedIterationsAndShiftsAccumulator acc(data_capacity);
      build_node_implicit_from_existing(node, left, right, num_actual_keys,
                                        data_capacity, &acc, &model);
      expected_avg_exp_search_iterations =
          acc.get_expected_num_search_iterations();
      expected_avg_shifts = acc.get_expected_num_shifts();
    }
    cost = kExpSearchIterationsWeight * expected_avg_exp_search_iterations +
           kShiftsWeight * expected_avg_shifts * expected_insert_frac;

    if (stats) {
      stats->num_search_iterations = expected_avg_exp_search_iterations;
      stats->num_shifts = expected_avg_shifts;
    }

    return cost;
  }

  // Helper function for compute_expected_cost
  // Implicitly build the data node in order to collect the stats
  static void build_node_implicit_from_existing(const self_type* node, int left,
                                                int right, int num_actual_keys,
                                                int data_capacity,
                                                StatAccumulator* acc,
                                                const LinearModel* model) {
    int last_position = -1;
    int keys_remaining = num_actual_keys;
    const_iterator_type it(node, left);
    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int predicted_position =
          std::max(0, std::min(data_capacity - 1, model->predict(it.key())));
      int actual_position =
          std::max<int>(predicted_position, last_position + 1);
      int positions_remaining = data_capacity - actual_position;
      if (positions_remaining < keys_remaining) {
        actual_position = data_capacity - keys_remaining;
        for (; actual_position < data_capacity; actual_position++, it++) {
          predicted_position = std::max(
              0, std::min(data_capacity - 1, model->predict(it.key())));
          acc->accumulate(actual_position, predicted_position);
        }
        break;
      }
      acc->accumulate(actual_position, predicted_position);
      last_position = actual_position;
      keys_remaining--;
    }
  }

  /*** Bulk loading and model building ***/

  // Initalize key/payload/bitmap arrays and relevant metadata
  void initialize(int num_keys, double density) {
    num_keys_ = num_keys;
    data_capacity_ =
        std::max(static_cast<int>(num_keys / density), num_keys + 1);
    bitmap_size_ = static_cast<size_t>(std::ceil(data_capacity_ / 64.));
    bitmap_ = new (bitmap_allocator().allocate(bitmap_size_))
        uint64_t[bitmap_size_]();  // initialize to all false
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ =
        new (key_allocator().allocate(data_capacity_)) AlexKey[data_capacity_];
    payload_slots_ =
        new (payload_allocator().allocate(data_capacity_)) P[data_capacity_];
#else
    data_slots_ =
        new (value_allocator().allocate(data_capacity_)) V[data_capacity];
#endif
  }

  // Assumes pretrained_model is trained on dense array of keys
  // I also assumed that all DataNodes have properly initialized key length limit. (max_key_length_)
  // second condition must be handled when creating data node.
  void bulk_load(const V values[], int num_keys,
                 const LinearModel* pretrained_model = nullptr,
                 bool train_with_sample = false) {
    /* minimal condition checking. */
    if (num_keys != 0) {assert(values[0].first.max_key_length_ == max_key_length_);}
    initialize(num_keys, kInitDensity_);

    if (num_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      for (int i = 0; i < data_capacity_; i++) {
        ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
      }
      return;
    }

    // Build model
    if (pretrained_model != nullptr) {
      assert(pretrained_model->max_key_length_ == max_key_length_);
      this->model_.max_key_length_ = pretrained_model->max_key_length_;
      this->model_.a_ = new double[pretrained_model->max_key_length_];
      for (unsigned int i = 0; i < pretrained_model->max_key_length_; i++) {
        this->model_.a_[i] = pretrained_model->a_[i];
      }
      this->model_.b_ = pretrained_model->b_;
    } else {
      build_model(values, num_keys, &(this->model_), train_with_sample);
    }
    this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

    // Model-based inserts
    int last_position = -1;
    int keys_remaining = num_keys;
    for (int i = 0; i < num_keys; i++) {
      int position = this->model_.predict(values[i].first);
      position = std::max<int>(position, last_position + 1);

      int positions_remaining = data_capacity_ - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = data_capacity_ - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
          ALEX_DATA_NODE_KEY_AT(j) = values[i].first;
        }
        for (int j = i; j < num_keys; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          key_slots_[pos] = values[j].first;
          payload_slots_[pos] = values[j].second;
#else
          data_slots_[pos] = values[j];
#endif
          set_bit(pos);
          pos++;
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
        ALEX_DATA_NODE_KEY_AT(j) = values[i].first;
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      key_slots_[position] = values[i].first;
      payload_slots_[position] = values[i].second;
#else
      data_slots_[position] = values[i];
#endif
      set_bit(position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < data_capacity_; i++) {
      ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
    }

    expansion_threshold_ = std::min(std::max(data_capacity_ * kMaxDensity_,
                                             static_cast<double>(num_keys + 1)),
                                    static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;

    for (unsigned int i = 0; i < this->max_key_length_; i++) {
       min_key_->key_arr_[i] = values[0].first.key_arr_[i];
       max_key_->key_arr_[i] = values[num_keys-1].first.key_arr_[i];
       mid_key_->key_arr_[i] = values[num_keys/2].first.key_arr_[i];
    }

    //updating Mnode_min_key_ / max_ for all parents of current data node
    AlexModelNode<P, Alloc> *cur_parent = this->parent_;
    while (cur_parent != nullptr) {
      char no_chg = 1;
      if (cur_parent->Mnode_min_key_ == nullptr) {
        cur_parent->Mnode_min_key_ = new double[cur_parent->max_key_length_];
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (unsigned int i = 0; i < this->max_key_length_; i++) {
          if (min_key_->key_arr_[i] < cur_parent->Mnode_min_key_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
          }
        }
      }
      if (cur_parent->Mnode_max_key_ == nullptr) {
        cur_parent->Mnode_max_key_ = new double[cur_parent->max_key_length_];
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (unsigned int i = 0; i < this->max_key_length_; i++) {
          if (cur_parent->Mnode_max_key_[i] < max_key_->key_arr_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
          }
        }
      }
      if (no_chg) {break;}
    }
  }

  // Bulk load using the keys between the left and right positions in
  // key/data_slots of an existing data node
  // keep_left and keep_right are set if the existing node was append-mostly
  // If the linear model and num_actual_keys have been precomputed, we can avoid
  // redundant work
  void bulk_load_from_existing(
      const self_type* node, int left, int right, bool keep_left = false,
      bool keep_right = false,
      const LinearModel* precomputed_model = nullptr,
      int precomputed_num_actual_keys = -1) {
    assert(left >= 0 && right <= node->data_capacity_);
    assert(node->max_key_length_ == max_key_length_);

    // Build model
    int num_actual_keys = 0;
    if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
      const_iterator_type it(node, left);
      LinearModelBuilder builder(&(this->model_));
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      assert(precomputed_model->max_key_length_ == max_key_length_);
      num_actual_keys = precomputed_num_actual_keys;
      this->model_.max_key_length_ = precomputed_model->max_key_length_;
      this->model_.a_ = new double[precomputed_model->max_key_length_];
      for (unsigned int i = 0; i < precomputed_model->max_key_length_; i++) {
        this->model_.a_[i] = precomputed_model->a_[i];
      }
      this->model_.b_ = precomputed_model->b_;
    }

    initialize(num_actual_keys, kMinDensity_);
    if (num_actual_keys == 0) {
      expansion_threshold_ = data_capacity_;
      contraction_threshold_ = 0;
      for (int i = 0; i < data_capacity_; i++) {
        ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
      }
      return;
    }

    // Special casing if existing node was append-mostly
    if (keep_left) {
      this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
    } else if (keep_right) {
      this->model_.expand((num_actual_keys / kMaxDensity_) / num_keys_);
      this->model_.b_ += (data_capacity_ - (num_actual_keys / kMaxDensity_));
    } else {
      this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
    }

    // Model-based inserts
    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(node, left);
    for (unsigned int i = 0; i < this->max_key_length_; i++) {
      min_key_->key_arr_[i] = it.key().key_arr_[i];
    }
    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int position = this->model_.predict(it.key());
      position = std::max<int>(position, last_position + 1);

      //mid_key update : first time when it is above the mid boundary.
      if (position >= data_capacity_/2) {
        for (unsigned int i = 0; i < this->max_key_length_; i++) {
          mid_key_->key_arr_[i] = it.key().key_arr_[i];
        }
      }

      int positions_remaining = data_capacity_ - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = data_capacity_ - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
          ALEX_DATA_NODE_KEY_AT(j) = it.key();
        }
        for (; pos < data_capacity_; pos++, it++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          key_slots_[pos] = it.key();
          payload_slots_[pos] = it.payload();
#else
          data_slots_[pos] = *it;
#endif
          set_bit(pos);
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
        ALEX_DATA_NODE_KEY_AT(j) = it.key();
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      key_slots_[position] = it.key();
      payload_slots_[position] = it.payload();
#else
      data_slots_[position] = *it;
#endif
      set_bit(position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < data_capacity_; i++) {
      ALEX_DATA_NODE_KEY_AT(i) = kEndSentinel_;
    }

    for (unsigned int i = 0; i < this->max_key_length_; i++) {
      max_key_->key_arr_[i] = ALEX_DATA_NODE_KEY_AT(last_position).key_arr_[i];
    }

    expansion_threshold_ =
        std::min(std::max(data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;

    AlexModelNode<P, Alloc> *cur_parent = this->parent;
    while (cur_parent != nullptr) {
      char no_chg = 1;
      if (cur_parent->Mnode_min_key_ == nullptr) {
        cur_parent->Mnode_min_key_ = new double[cur_parent->max_key_length_];
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (unsigned int i = 0; i < this->max_key_length_; i++) {
          if (min_key_->key_arr_[i] < cur_parent->Mnode_min_key_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
          }
        }
      }
      if (cur_parent->Mnode_max_key_ == nullptr) {
        cur_parent->Mnode_max_key_ = new double[cur_parent->max_key_length_];
        for (int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (int i = 0; i < cur_parent->max_key_length_; i++) {
          if (cur_parent->Mnode_max_key_[i] < max_key_->key_arr_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
          }
        }
      }
      if (no_chg) {break;}
    }
  }

  static void build_model(const V* values, int num_keys, LinearModel* model,
                          bool use_sampling = false) {
    /* sampling only possible for integer, double type keys... for now.*/
    if (use_sampling && (model->max_key_length_ == 1)) {
      build_model_sampling(values, num_keys, model);
      return;
    }

    LinearModelBuilder builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  // Uses progressive non-random uniform sampling to build the model
  // Progressively increases sample size until model parameters are relatively
  // stable
  static void build_model_sampling(const V* values, int num_keys,
                                   LinearModel* model,
                                   bool verbose = false) {
    const static int sample_size_lower_bound = 10;
    // If slope and intercept change by less than this much between samples,
    // return
    const static double rel_change_threshold = 0.01;
    // If intercept changes by less than this much between samples, return
    const static double abs_change_threshold = 0.5;
    // Increase sample size by this many times each iteration
    const static int sample_size_multiplier = 2;

    // If the number of keys is sufficiently small, we do not sample
    if (num_keys <= sample_size_lower_bound * sample_size_multiplier) {
      build_model(values, num_keys, model, false);
      return;
    }

    int step_size = 1;
    double sample_size = num_keys;
    while (sample_size >= sample_size_lower_bound) {
      sample_size /= sample_size_multiplier;
      step_size *= sample_size_multiplier;
    }
    step_size /= sample_size_multiplier;

    // Run with initial step size
    LinearModelBuilder builder(model);
    for (int i = 0; i < num_keys; i += step_size) {
      builder.add(values[i].first, i);
    }
    builder.build();
    double prev_a[model->max_key_length_] = {0.0};
    for (unsigned int i = 0; i < model->max_key_length_; i++) {
      prev_a[i] = model->a_[i];
    }
    double prev_b = model->b_;
    if (verbose) {
      std::cout << "Build index, sample size: " << num_keys / step_size
                << " a: (";
      for (unsigned int i = 0; i < model->max_key_length_; i++) {
        std::cout << prev_a[i];
      } 
      std::cout << "), b: " << prev_b << std::endl;
    }

    // Keep decreasing step size (increasing sample size) until model does not
    // change significantly
    while (step_size > 1) {
      step_size /= sample_size_multiplier;
      // Need to avoid processing keys we already processed in previous samples
      int i = 0;
      while (i < num_keys) {
        i += step_size;
        for (int j = 1; (j < sample_size_multiplier) && (i < num_keys);
             j++, i += step_size) {
          builder.add(values[i].first, i);
        }
      }
      builder.build();

      double rel_change_in_a[model->max_key_length_] = {0.0};
      for (unsigned int i = 0; i < model->max_key_length_; i++) {
        rel_change_in_a[i] = std::abs((model->a_[i] - prev_a[i]) / prev_a[i]);
      }
      double abs_change_in_b = std::abs(model->b_ - prev_b);
      double rel_change_in_b = std::abs(abs_change_in_b / prev_b);
      if (verbose) {
        std::cout << "Build index, sample size: " << num_keys / step_size
                  << " new (a, b): (";
        for (unsigned int i = 0; i < model->max_key_length_; i++) {
          std::cout << model->a_[i];
        }
        std::cout << ", " << model->b_ << ") relative change : (";
        for (unsigned int i = 0; i < model->max_key_length_; i++) {
          std::cout << rel_change_in_a[i];
        }
        std::cout << ", " << rel_change_in_b << ")"
                  << std::endl;
      }
      char threshold = 1;
      for (unsigned int i = 0; i < model->max_key_length_; i++) {
        if (rel_change_in_a[i] > rel_change_threshold) {
          threshold = 0;
          break;
        }
      }
      if (threshold &&
          (rel_change_in_b < rel_change_threshold ||
           abs_change_in_b < abs_change_threshold)) {
        return;
      }
      for (unsigned int i = 0; i < model->max_key_length_; i++) {
        prev_a[i] = model->a_[i];
      }
      prev_b = model->b_;
    }
  }

  // Unused function: builds a spline model by connecting the smallest and
  // largest points instead of using
  // a linear regression
  //static void build_spline(const V* values, int num_keys,
  //                         const LinearModel<T>* model) {
  //  int y_max = num_keys - 1;
  //  int y_min = 0;
  //  model->a_ = static_cast<double>(y_max - y_min) /
  //              (values[y_max].first - values[y_min].first);
  //  model->b_ = -1.0 * values[y_min].first * model->a_;
  //}

  /*** Lookup ***/

  // Predicts the position of a key using the model
  inline int predict_position(const AlexKey& key) const {
    int position = this->model_.predict(key);
    position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
    return position;
  }

  // Searches for the last non-gap position equal to key
  // If no positions equal to key, returns -1
  int find_key(const AlexKey& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    // The last key slot with a certain value is guaranteed to be a real key
    // (instead of a gap)
    int pos = exponential_search_upper_bound(predicted_pos, key) - 1;
    if (pos < 0 || !key_equal(ALEX_DATA_NODE_KEY_AT(pos), key)) {
      return -1;
    } else {
      return pos;
    }
  }

  // Searches for the first non-gap position no less than key
  // Returns position in range [0, data_capacity]
  // Compare with lower_bound()
  int find_lower(const AlexKey& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    int pos = exponential_search_lower_bound(predicted_pos, key);
    return get_next_filled_position(pos, false);
  }

  // Searches for the first non-gap position greater than key
  // Returns position in range [0, data_capacity]
  // Compare with upper_bound()
  int find_upper(const AlexKey& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    int pos = exponential_search_upper_bound(predicted_pos, key);
    return get_next_filled_position(pos, false);
  }

  // Finds position to insert a key.
  // First returned value takes prediction into account.
  // Second returned value is first valid position (i.e., upper_bound of key).
  // If there are duplicate keys, the insert position will be to the right of
  // all existing keys of the same value.
  std::pair<int, int> find_insert_position(const AlexKey& key) {
    int predicted_pos =
        predict_position(key);  // first use model to get prediction

    // insert to the right of duplicate keys
    int pos = exponential_search_upper_bound(predicted_pos, key);
    if (predicted_pos <= pos || check_exists(pos)) {
      return {pos, pos};
    } else {
      // Place inserted key as close as possible to the predicted position while
      // maintaining correctness
      return {std::min(predicted_pos, get_next_filled_position(pos, true) - 1),
              pos};
    }
  }

  // Starting from a position, return the first position that is not a gap
  // If no more filled positions, will return data_capacity
  // If exclusive is true, output is at least (pos + 1)
  // If exclusive is false, output can be pos itself
  int get_next_filled_position(int pos, bool exclusive) const {
    if (exclusive) {
      pos++;
      if (pos == data_capacity_) {
        return data_capacity_;
      }
    }

    int curBitmapIdx = pos >> 6;
    uint64_t curBitmapData = bitmap_[curBitmapIdx];

    // Zero out extra bits
    int bit_pos = pos - (curBitmapIdx << 6);
    curBitmapData &= ~((1ULL << (bit_pos)) - 1);

    while (curBitmapData == 0) {
      curBitmapIdx++;
      if (curBitmapIdx >= bitmap_size_) {
        return data_capacity_;
      }
      curBitmapData = bitmap_[curBitmapIdx];
    }
    uint64_t bit = extract_rightmost_one(curBitmapData);
    return get_offset(curBitmapIdx, bit);
  }

  // Searches for the first position greater than key
  // This could be the position for a gap (i.e., its bit in the bitmap is 0)
  // Returns position in range [0, data_capacity]
  // Compare with find_upper()
  int upper_bound(const AlexKey& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_upper_bound(position, key);
  }

  // Searches for the first position greater than key, starting from position m
  // Returns position in range [0, data_capacity]
  inline int exponential_search_upper_bound(int m, const AlexKey& key) {
    // Continue doubling the bound until it contains the upper bound. Then use
    // binary search.
    int bound = 1;
    int l, r;  // will do binary search in range [l, r)
    if (key_greater(ALEX_DATA_NODE_KEY_AT(m), key)) {
      int size = m;
      while (bound < size &&
             key_greater(ALEX_DATA_NODE_KEY_AT(m - bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m - std::min<int>(bound, size);
      r = m - bound / 2;
    } else {
      int size = data_capacity_ - m;
      while (bound < size &&
             key_lessequal(ALEX_DATA_NODE_KEY_AT(m + bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m + bound / 2;
      r = m + std::min<int>(bound, size);
    }
    return binary_search_upper_bound(l, r, key);
  }

  // Searches for the first position greater than key in range [l, r)
  // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
  // Returns position in range [l, r]
  inline int binary_search_upper_bound(int l, int r, const AlexKey& key) const {
    while (l < r) {
      int mid = l + (r - l) / 2;
      if (key_lessequal(ALEX_DATA_NODE_KEY_AT(mid), key)) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    return l;
  }

  // Searches for the first position no less than key
  // This could be the position for a gap (i.e., its bit in the bitmap is 0)
  // Returns position in range [0, data_capacity]
  // Compare with find_lower()
  int lower_bound(const AlexKey& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_lower_bound(position, key);
  }

  // Searches for the first position no less than key, starting from position m
  // Returns position in range [0, data_capacity]
  inline int exponential_search_lower_bound(int m, const AlexKey& key) {
    // Continue doubling the bound until it contains the lower bound. Then use
    // binary search.
    int bound = 1;
    int l, r;  // will do binary search in range [l, r)
    if (key_greaterequal(ALEX_DATA_NODE_KEY_AT(m), key)) {
      int size = m;
      while (bound < size &&
             key_greaterequal(ALEX_DATA_NODE_KEY_AT(m - bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m - std::min<int>(bound, size);
      r = m - bound / 2;
    } else {
      int size = data_capacity_ - m;
      while (bound < size && key_less(ALEX_DATA_NODE_KEY_AT(m + bound), key)) {
        bound *= 2;
        num_exp_search_iterations_++;
      }
      l = m + bound / 2;
      r = m + std::min<int>(bound, size);
    }
    return binary_search_lower_bound(l, r, key);
  }

  // Searches for the first position no less than key in range [l, r)
  // https://stackoverflow.com/questions/6443569/implementation-of-c-lower-bound
  // Returns position in range [l, r]
  inline int binary_search_lower_bound(int l, int r, const AlexKey& key) const {
    while (l < r) {
      int mid = l + (r - l) / 2;
      if (key_greaterequal(ALEX_DATA_NODE_KEY_AT(mid), key)) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    return l;
  }

  /*** Inserts and resizes ***/

  // Whether empirical cost deviates significantly from expected cost
  // Also returns false if empirical cost is sufficiently low and is not worth
  // splitting
  inline bool significant_cost_deviation() const {
    double emp_cost = empirical_cost();
    return emp_cost > kNodeLookupsWeight && emp_cost > 1.5 * this->cost_;
  }

  // Returns true if cost is catastrophically high and we want to force a split
  // The heuristic for this is if the number of shifts per insert (expected or
  // empirical) is over 100
  inline bool catastrophic_cost() const {
    return shifts_per_insert() > 100 || expected_avg_shifts_ > 100;
  }

  // First value in returned pair is fail flag:
  // 0 if successful insert (possibly with automatic expansion).
  // 1 if no insert because of significant cost deviation.
  // 2 if no insert because of "catastrophic" cost.
  // 3 if no insert because node is at max capacity.
  // -1 if key already exists and duplicates not allowed.
  //
  // Second value in returned pair is position of inserted key, or of the
  // already-existing key.
  // -1 if no insertion.
  std::pair<int, int> insert(const AlexKey& key, const P& payload) {
    // Periodically check for catastrophe
    if (num_inserts_ % 64 == 0 && catastrophic_cost()) {
      return {2, -1};
    }

    // Check if node is full (based on expansion_threshold)
    if (num_keys_ >= expansion_threshold_) {
      if (significant_cost_deviation()) {
        return {1, -1};
      }
      if (catastrophic_cost()) {
        return {2, -1};
      }
      if (num_keys_ > max_slots_ * kMinDensity_) {
        return {3, -1};
      }
      // Expand
      bool keep_left = is_append_mostly_right();
      bool keep_right = is_append_mostly_left();
      resize(kMinDensity_, false, keep_left, keep_right);
      num_resizes_++;
    }

    // Insert
    std::pair<int, int> positions = find_insert_position(key);
    int upper_bound_pos = positions.second;
    if (!allow_duplicates && upper_bound_pos > 0 &&
        key_equal(ALEX_DATA_NODE_KEY_AT(upper_bound_pos - 1), key)) {
      return {-1, upper_bound_pos - 1};
    }
    int insertion_position = positions.first;
    if (insertion_position < data_capacity_ &&
        !check_exists(insertion_position)) {
      insert_element_at(key, payload, insertion_position);
    } else {
      insertion_position =
          insert_using_shifts(key, payload, insertion_position);
    }

    // Update stats
    num_keys_++;
    num_inserts_++;
    if (key_less_(*max_key_, key)) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        max_key_->key_arr_[i] = key.key_arr_[i];
      }
      num_right_out_of_bounds_inserts_++;
    }
    if (key_less_(key, *min_key_)) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        min_key_->key_arr_[i] = key.key_arr_[i];
      }
      num_left_out_of_bounds_inserts_++;
    }
    char mid_chg = 0, not_kEnd = 0;
#if ALEX_DATA_NODE_SEP_ARRAYS
    for (unsigned int i = 0; i < this->max_key_length_; i++) {
      if (mid_key_->key_arr_[i] != key_slots_[data_capacity_/2].key_arr_[i]) {
        mid_chg = 1;
        break;
      }
    }
    if (mid_chg) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        if (key_slots_[data_capacity_/2].key_arr_[i] != kEndSentinel_.key_arr_[i]) {
          not_kEnd = 1;
          break;
        }
      }
    }
    if (not_kEnd) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        mid_key_->key_arr_[i] = key_slots_[data_capacity_/2].key_arr_[i];
      }
    }
#else
    for (unsigned int i = 0; i < this->max_key_length_; i++) {
      if (mid_key_->key_arr_[i] != data_slots_[data_capacith_/2].first.key_arr_[i]) {
        mid_chg = 1;
        break;
      }
    }
    if (mid_chg) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        if (data_slots_[data_capacity_/2].first.key_arr_[i] != kEndSentinel->key_arr_[i]) {
          not_kEnd = 1;
          break;
        }
      }
    }
    if (not_kEnd) {
      for (unsigned int i = 0; i < this->max_key_length_; i++) {
        mid_key_->key_arr_[i] = data_slots_[data_capacity_/2].first.key_arr_[i];
      }
    }
#endif

    AlexModelNode<P, Alloc> *cur_parent = this->parent_;
    while (cur_parent != nullptr) {
      char no_chg = 1;
      if (cur_parent->Mnode_min_key_ == nullptr) {
        cur_parent->Mnode_min_key_ = new double[cur_parent->max_key_length_];
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (unsigned int i = 0; i < this->max_key_length_; i++) {
          if (min_key_->key_arr_[i] < cur_parent->Mnode_min_key_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_min_key_[i] = min_key_->key_arr_[i];
          }
        }
      }
      if (cur_parent->Mnode_max_key_ == nullptr) {
        cur_parent->Mnode_max_key_ = new double[cur_parent->max_key_length_];
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
        }
        no_chg = 0;
      }
      else {
        char need_chg = 0;
        for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
          if (cur_parent->Mnode_max_key_[i] < max_key_->key_arr_[i]) {
            need_chg = 1;
            no_chg = 0;
            break;
          }
        }
        if (need_chg) {
          for (unsigned int i = 0; i < cur_parent->max_key_length_; i++) {
            cur_parent->Mnode_max_key_[i] = max_key_->key_arr_[i];
          }
        }
      }
      if (no_chg) {break;}
    }
    
    return {0, insertion_position};
  }

  // Resize the data node to the target density
  void resize(double target_density, bool force_retrain = false,
              bool keep_left = false, bool keep_right = false) {
    if (num_keys_ == 0) {
      return;
    }

    int new_data_capacity =
        std::max(static_cast<int>(num_keys_ / target_density), num_keys_ + 1);
    auto new_bitmap_size =
        static_cast<size_t>(std::ceil(new_data_capacity / 64.));
    auto new_bitmap = new (bitmap_allocator().allocate(new_bitmap_size))
        uint64_t[new_bitmap_size]();  // initialize to all false
#if ALEX_DATA_NODE_SEP_ARRAYS
    AlexKey* new_key_slots =
        new (key_allocator().allocate(new_data_capacity)) AlexKey[new_data_capacity];
    P* new_payload_slots = new (payload_allocator().allocate(new_data_capacity))
        P[new_data_capacity];
#else
    V* new_data_slots = new (value_allocator().allocate(new_data_capacity))
        V[new_data_capacity];
#endif

    // Retrain model if the number of keys is sufficiently small (under 50)
    if (num_keys_ < 50 || force_retrain) {
      const_iterator_type it(this, 0);
      LinearModelBuilder builder(&(this->model_));
      for (int i = 0; it.cur_idx_ < data_capacity_ && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
      }
      builder.build();
      if (keep_left) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      } else if (keep_right) {
        this->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else {
        this->model_.expand(static_cast<double>(new_data_capacity) / num_keys_);
      }
    } else {
      if (keep_right) {
        this->model_.b_ += (new_data_capacity - data_capacity_);
      } else if (!keep_left) {
        this->model_.expand(static_cast<double>(new_data_capacity) /
                            data_capacity_);
      }
    }

    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(this, 0);
    for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
      int position = this->model_.predict(it.key());
      position = std::max<int>(position, last_position + 1);

      int positions_remaining = new_data_capacity - position;
      if (positions_remaining < keys_remaining) {
        // fill the rest of the store contiguously
        int pos = new_data_capacity - keys_remaining;
        for (int j = last_position + 1; j < pos; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          new_key_slots[j] = it.key();
#else
          new_data_slots[j].first = it.key();
#endif
        }
        for (; pos < new_data_capacity; pos++, it++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
          new_key_slots[pos] = it.key();
          new_payload_slots[pos] = it.payload();
#else
          new_data_slots[pos] = *it;
#endif
          set_bit(new_bitmap, pos);
        }
        last_position = pos - 1;
        break;
      }

      for (int j = last_position + 1; j < position; j++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        new_key_slots[j] = it.key();
#else
        new_data_slots[j].first = it.key();
#endif
      }

#if ALEX_DATA_NODE_SEP_ARRAYS
      new_key_slots[position] = it.key();
      new_payload_slots[position] = it.payload();
#else
      new_data_slots[position] = *it;
#endif
      set_bit(new_bitmap, position);

      last_position = position;

      keys_remaining--;
    }

    for (int i = last_position + 1; i < new_data_capacity; i++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
      new_key_slots[i] = kEndSentinel_;
#else
      new_data_slots[i].first = kEndSentinel;
#endif
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    key_allocator().deallocate(key_slots_, data_capacity_);
    payload_allocator().deallocate(payload_slots_, data_capacity_);
#else
    value_allocator().deallocate(data_slots_, data_capacity_);
#endif
    bitmap_allocator().deallocate(bitmap_, bitmap_size_);

    data_capacity_ = new_data_capacity;
    bitmap_size_ = new_bitmap_size;
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new_key_slots;
    payload_slots_ = new_payload_slots;
#else
    data_slots_ = new_data_slots;
#endif
    bitmap_ = new_bitmap;

    expansion_threshold_ =
        std::min(std::max(data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;
  }

  inline bool is_append_mostly_right() const {
    return static_cast<double>(num_right_out_of_bounds_inserts_) /
               num_inserts_ >
           kAppendMostlyThreshold;
  }

  inline bool is_append_mostly_left() const {
    return static_cast<double>(num_left_out_of_bounds_inserts_) / num_inserts_ >
           kAppendMostlyThreshold;
  }

  // Insert key into pos. The caller must guarantee that pos is a gap.
  void insert_element_at(const AlexKey& key, P payload, int pos) {
#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_[pos] = key;
    payload_slots_[pos] = payload;
#else
    data_slots_[index] = std::make_pair(key, payload);
#endif
    set_bit(pos);

    // Overwrite preceding gaps until we reach the previous element
    pos--;
    while (pos >= 0 && !check_exists(pos)) {
      ALEX_DATA_NODE_KEY_AT(pos) = key;
      pos--;
    }
  }

  // Insert key into pos, shifting as necessary in the range [left, right)
  // Returns the actual position of insertion
  int insert_using_shifts(const AlexKey& key, P payload, int pos) {
    // Find the closest gap
    int gap_pos = closest_gap(pos);
    set_bit(gap_pos);
    if (gap_pos >= pos) {
      for (int i = gap_pos; i > pos; i--) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        key_slots_[i] = key_slots_[i - 1];
        payload_slots_[i] = payload_slots_[i - 1];
#else
        data_slots_[i] = data_slots_[i - 1];
#endif
      }
      insert_element_at(key, payload, pos);
      num_shifts_ += gap_pos - pos;
      return pos;
    } else {
      for (int i = gap_pos; i < pos - 1; i++) {
#if ALEX_DATA_NODE_SEP_ARRAYS
        key_slots_[i] = key_slots_[i + 1];
        payload_slots_[i] = payload_slots_[i + 1];
#else
        data_slots_[i] = data_slots_[i + 1];
#endif
      }
      insert_element_at(key, payload, pos - 1);
      num_shifts_ += pos - gap_pos - 1;
      return pos - 1;
    }
  }

#if ALEX_USE_LZCNT
  // Returns position of closest gap to pos
  // Returns pos if pos is a gap
  int closest_gap(int pos) const {
    pos = std::min(pos, data_capacity_ - 1);
    int bitmap_pos = pos >> 6;
    int bit_pos = pos - (bitmap_pos << 6);
    if (bitmap_[bitmap_pos] == static_cast<uint64_t>(-1) ||
        (bitmap_pos == bitmap_size_ - 1 &&
         _mm_popcnt_u64(bitmap_[bitmap_pos]) ==
             data_capacity_ - ((bitmap_size_ - 1) << 6))) {
      // no gaps in this block of 64 positions, start searching in adjacent
      // blocks
      int left_bitmap_pos = 0;
      int right_bitmap_pos = ((data_capacity_ - 1) >> 6);  // inclusive
      int max_left_bitmap_offset = bitmap_pos - left_bitmap_pos;
      int max_right_bitmap_offset = right_bitmap_pos - bitmap_pos;
      int max_bidirectional_bitmap_offset =
          std::min<int>(max_left_bitmap_offset, max_right_bitmap_offset);
      int bitmap_distance = 1;
      while (bitmap_distance <= max_bidirectional_bitmap_offset) {
        uint64_t left_bitmap_data = bitmap_[bitmap_pos - bitmap_distance];
        uint64_t right_bitmap_data = bitmap_[bitmap_pos + bitmap_distance];
        if (left_bitmap_data != static_cast<uint64_t>(-1) &&
            right_bitmap_data != static_cast<uint64_t>(-1)) {
          int left_gap_pos = ((bitmap_pos - bitmap_distance + 1) << 6) -
                             static_cast<int>(_lzcnt_u64(~left_bitmap_data)) -
                             1;
          int right_gap_pos = ((bitmap_pos + bitmap_distance) << 6) +
                              static_cast<int>(_tzcnt_u64(~right_bitmap_data));
          if (pos - left_gap_pos <= right_gap_pos - pos ||
              right_gap_pos >= data_capacity_) {
            return left_gap_pos;
          } else {
            return right_gap_pos;
          }
        } else if (left_bitmap_data != static_cast<uint64_t>(-1)) {
          int left_gap_pos = ((bitmap_pos - bitmap_distance + 1) << 6) -
                             static_cast<int>(_lzcnt_u64(~left_bitmap_data)) -
                             1;
          // also need to check next block to the right
          if (bit_pos > 32 && bitmap_pos + bitmap_distance + 1 < bitmap_size_ &&
              bitmap_[bitmap_pos + bitmap_distance + 1] !=
                  static_cast<uint64_t>(-1)) {
            int right_gap_pos =
                ((bitmap_pos + bitmap_distance + 1) << 6) +
                static_cast<int>(
                    _tzcnt_u64(~bitmap_[bitmap_pos + bitmap_distance + 1]));
            if (pos - left_gap_pos <= right_gap_pos - pos ||
                right_gap_pos >= data_capacity_) {
              return left_gap_pos;
            } else {
              return right_gap_pos;
            }
          } else {
            return left_gap_pos;
          }
        } else if (right_bitmap_data != static_cast<uint64_t>(-1)) {
          int right_gap_pos = ((bitmap_pos + bitmap_distance) << 6) +
                              static_cast<int>(_tzcnt_u64(~right_bitmap_data));
          if (right_gap_pos < data_capacity_) {
            // also need to check next block to the left
            if (bit_pos < 32 && bitmap_pos - bitmap_distance > 0 &&
                bitmap_[bitmap_pos - bitmap_distance - 1] !=
                    static_cast<uint64_t>(-1)) {
              int left_gap_pos =
                  ((bitmap_pos - bitmap_distance) << 6) -
                  static_cast<int>(
                      _lzcnt_u64(~bitmap_[bitmap_pos - bitmap_distance - 1])) -
                  1;
              if (pos - left_gap_pos <= right_gap_pos - pos ||
                  right_gap_pos >= data_capacity_) {
                return left_gap_pos;
              } else {
                return right_gap_pos;
              }
            } else {
              return right_gap_pos;
            }
          }
        }
        bitmap_distance++;
      }
      if (max_left_bitmap_offset > max_right_bitmap_offset) {
        for (int i = bitmap_pos - bitmap_distance; i >= left_bitmap_pos; i--) {
          if (bitmap_[i] != static_cast<uint64_t>(-1)) {
            return ((i + 1) << 6) - static_cast<int>(_lzcnt_u64(~bitmap_[i])) -
                   1;
          }
        }
      } else {
        for (int i = bitmap_pos + bitmap_distance; i <= right_bitmap_pos; i++) {
          if (bitmap_[i] != static_cast<uint64_t>(-1)) {
            int right_gap_pos =
                (i << 6) + static_cast<int>(_tzcnt_u64(~bitmap_[i]));
            if (right_gap_pos >= data_capacity_) {
              return -1;
            } else {
              return right_gap_pos;
            }
          }
        }
      }
      return -1;
    } else {
      // search within block of 64 positions
      uint64_t bitmap_data = bitmap_[bitmap_pos];
      int closest_right_gap_distance = 64;
      int closest_left_gap_distance = 64;
      // Logically gaps to the right of pos, in the bitmap these are gaps to the
      // left of pos's bit
      // This covers the case where pos is a gap
      // For example, if pos is 3, then bitmap '10101101' -> bitmap_right_gaps
      // '01010000'
      uint64_t bitmap_right_gaps = ~(bitmap_data | ((1ULL << bit_pos) - 1));
      if (bitmap_right_gaps != 0) {
        closest_right_gap_distance =
            static_cast<int>(_tzcnt_u64(bitmap_right_gaps)) - bit_pos;
      } else if (bitmap_pos + 1 < bitmap_size_) {
        // look in the next block to the right
        closest_right_gap_distance =
            64 + static_cast<int>(_tzcnt_u64(~bitmap_[bitmap_pos + 1])) -
            bit_pos;
      }
      // Logically gaps to the left of pos, in the bitmap these are gaps to the
      // right of pos's bit
      // For example, if pos is 3, then bitmap '10101101' -> bitmap_left_gaps
      // '00000010'
      uint64_t bitmap_left_gaps = (~bitmap_data) & ((1ULL << bit_pos) - 1);
      if (bitmap_left_gaps != 0) {
        closest_left_gap_distance =
            bit_pos - (63 - static_cast<int>(_lzcnt_u64(bitmap_left_gaps)));
      } else if (bitmap_pos > 0) {
        // look in the next block to the left
        closest_left_gap_distance =
            bit_pos + static_cast<int>(_lzcnt_u64(~bitmap_[bitmap_pos - 1])) +
            1;
      }

      if (closest_right_gap_distance < closest_left_gap_distance &&
          pos + closest_right_gap_distance < data_capacity_) {
        return pos + closest_right_gap_distance;
      } else {
        return pos - closest_left_gap_distance;
      }
    }
  }
#else
  // A slower version of closest_gap that does not use lzcnt and tzcnt
  // Does not return pos if pos is a gap
  int closest_gap(int pos) const {
    int max_left_offset = pos;
    int max_right_offset = data_capacity_ - pos - 1;
    int max_bidirectional_offset =
        std::min<int>(max_left_offset, max_right_offset);
    int distance = 1;
    while (distance <= max_bidirectional_offset) {
      if (!check_exists(pos - distance)) {
        return pos - distance;
      }
      if (!check_exists(pos + distance)) {
        return pos + distance;
      }
      distance++;
    }
    if (max_left_offset > max_right_offset) {
      for (int i = pos - distance; i >= 0; i--) {
        if (!check_exists(i)) return i;
      }
    } else {
      for (int i = pos + distance; i < data_capacity_; i++) {
        if (!check_exists(i)) return i;
      }
    }
    return -1;
  }
#endif

  /*** Deletes ***/

  // Erase the left-most key with the input value
  // Returns the number of keys erased (0 or 1)
  int erase_one(const AlexKey& key) {
    int pos = find_lower(key);

    if (pos == data_capacity_ || !key_equal(ALEX_DATA_NODE_KEY_AT(pos), key))
      return 0;

    // Erase key at pos
    erase_one_at(pos);
    return 1;
  }

  // Erase the key at the given position
  // CURRENTLY ASSUMING THAT DUPLICATE KEYS HAVE SEPARATE MEMORY... CHECK PLEASE (seems ok)
  void erase_one_at(int pos) {
    AlexKey next_key;
    if (pos == data_capacity_ - 1) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos + 1);
    }
    delete ALEX_DATA_NODE_KEY_AT(pos);
    ALEX_DATA_NODE_KEY_AT(pos) = next_key;
    unset_bit(pos);
    pos--;

    // Erase preceding gaps until we reach an existing key
    while (pos >= 0 && !check_exists(pos)) {
      delete ALEX_DATA_NODE_KEY_AT(pos);
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      pos--;
    }

    num_keys_--;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
  }

  // Erase all keys with the input value
  // Returns the number of keys erased (there may be multiple keys with the same
  // value)
  int erase(const AlexKey& key) {
    int pos = upper_bound(key);

    if (pos == 0 || !key_equal(ALEX_DATA_NODE_KEY_AT(pos - 1), key)) return 0;

    // Erase preceding positions until we reach a key with smaller value
    int num_erased = 0;
    AlexKey next_key;
    if (pos == data_capacity_) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos);
    }
    pos--;
    while (pos >= 0 && key_equal(ALEX_DATA_NODE_KEY_AT(pos), key)) {
      delete ALEX_DATA_NODE_KEY_AT(pos);
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      num_erased += check_exists(pos);
      unset_bit(pos);
      pos--;
    }

    num_keys_ -= num_erased;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
    return num_erased;
  }

  // Erase keys with value between start key (inclusive) and end key.
  // Returns the number of keys erased.
  int erase_range(AlexKey start_key, AlexKey end_key, bool end_key_inclusive = false) {
    int pos;
    if (end_key_inclusive) {
      pos = upper_bound(end_key);
    } else {
      pos = lower_bound(end_key);
    }

    if (pos == 0) return 0;

    // Erase preceding positions until key value is below the start key
    int num_erased = 0;
    AlexKey next_key;
    if (pos == data_capacity_) {
      next_key = kEndSentinel_;
    } else {
      next_key = ALEX_DATA_NODE_KEY_AT(pos);
    }
    pos--;
    while (pos >= 0 &&
           key_greaterequal(ALEX_DATA_NODE_KEY_AT(pos), start_key)) {
      delete ALEX_DATA_NODE_KEY_AT(pos);
      ALEX_DATA_NODE_KEY_AT(pos) = next_key;
      num_erased += check_exists(pos);
      unset_bit(pos);
      pos--;
    }

    num_keys_ -= num_erased;

    if (num_keys_ < contraction_threshold_) {
      resize(kMaxDensity_);  // contract
      num_resizes_++;
    }
    return num_erased;
  }

  /*** Stats ***/

  // Total size of node metadata
  long long node_size() const override { return sizeof(self_type); }

  // Total size in bytes of key/payload/data_slots and bitmap
  // NOTE THAT IT DOESN'T INCLUDE ALEX KEY'S POINTING ARRAY SIZE.
  long long data_size() const {
    long long data_size = data_capacity_ * sizeof(AlexKey);
    data_size += data_capacity_ * sizeof(P);
    data_size += bitmap_size_ * sizeof(uint64_t);
    return data_size;
  }

  // Number of contiguous blocks of keys without gaps
  int num_packed_regions() const {
    int num_packed = 0;
    bool is_packed = check_exists(0);
    for (int i = 1; i < data_capacity_; i++) {
      if (check_exists(i) != is_packed) {
        if (is_packed) {
          num_packed++;
        }
        is_packed = !is_packed;
      }
    }
    if (is_packed) {
      num_packed++;
    }
    return num_packed;
  }

  /*** Debugging ***/

  bool validate_structure(bool verbose = false) const {
    if (this->cost_ < 0 || std::isnan(this->cost_)) {
      std::cout << "[Data node cost is invalid value]"
                << " node addr: " << this << ", node level: " << this->level_
                << ", cost: " << this->cost_ << std::endl;
      return false;
    }
    for (int i = 0; i < data_capacity_ - 1; i++) {
      if (key_greater(ALEX_DATA_NODE_KEY_AT(i), ALEX_DATA_NODE_KEY_AT(i + 1))) {
        if (verbose) {
          std::cout << "Keys should be in non-increasing order" << std::endl;
        }
        return false;
      } else if (key_less(ALEX_DATA_NODE_KEY_AT(i),
                          ALEX_DATA_NODE_KEY_AT(i + 1)) &&
                 !check_exists(i)) {
        if (verbose) {
          std::cout << "The last key of a certain value should not be a gap"
                    << std::endl;
        }
        return false;
      }
    }
    AlexKey end = ALEX_DATA_NODE_KEY_AT(data_capacity_ - 1);
    char same = 0;
    for (int i = 0; i < this->max_key_length_; i++) {
      if (end.key_arr_[i] != kEndSentinel_.key_arr_[i]) {
        same = 1;
        break;
      }
    }
    if (same && check_exists(data_capacity_ - 1)) {
      if (verbose) {
        std::cout << "The sentinel should not be a valid key" << std::endl;
      }
      return false;
    }
    if (!same && !check_exists(data_capacity_ - 1)) {
      if (verbose) {
        std::cout << "The last key should be a valid key" << std::endl;
      }
      return false;
    }
    uint64_t num_bitmap_ones = 0;
    for (int i = 0; i < bitmap_size_; i++) {
      num_bitmap_ones += count_ones(bitmap_[i]);
    }
    if (static_cast<int>(num_bitmap_ones) != num_keys_) {
      if (verbose) {
        std::cout << "Number of ones in bitmap should match num_keys"
                  << std::endl;
      }
      return false;
    }
    return true;
  }

  // Check that a key exists in the key/data_slots
  // If validate_bitmap is true, confirm that the corresponding position in the
  // bitmap is correctly set to 1
  bool key_exists(const AlexKey& key, bool validate_bitmap) const {
    for (int i = 0; i < data_capacity_ - 1; i++) {
      if (key_equal(ALEX_DATA_NODE_KEY_AT(i), key) &&
          (!validate_bitmap || check_exists(i))) {
        return true;
      }
    }
    return false;
  }

  std::string to_string() const {
    std::string str;
    str += "Num keys: " + std::to_string(num_keys_) + ", Capacity: " +
           std::to_string(data_capacity_) + ", Expansion Threshold: " +
           std::to_string(expansion_threshold_) + "\n";
    for (int i = 0; i < data_capacity_; i++) {
      AlexKey cur_key = ALEX_DATA_NODE_KEY_AT(i);
      for (int j = 0; j < cur_key.max_key_length_; j++) {
        str += (std::to_string(cur_key.key_arr_[j]) + " ");
      }
      str += "\n";
    }
    return str;
  }
};
}
