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
#include <map>

// Whether we store key and payload arrays separately in data nodes
// By default, we store them separately
#define ALEX_DATA_NODE_SEP_ARRAYS 1

#if ALEX_DATA_NODE_SEP_ARRAYS
#define ALEX_DATA_NODE_KEY_AT(i) key_slots_[i]
#define ALEX_DATA_NODE_PAYLOAD_AT(i) payload_slots_[i]
#else
#define ALEX_DATA_NODE_KEY_AT(i) data_slots_[i].first
#define ALEX_DATA_NODE_PAYLOAD_AT(i) data_slots_[i].second.read()
#endif

// Whether we use lzcnt and tzcnt when manipulating a bitmap (e.g., when finding
// the closest gap).
// If your hardware does not support lzcnt/tzcnt (e.g., your Intel CPU is
// pre-Haswell), set this to 0.
#define ALEX_USE_LZCNT 1

namespace alex {

//forward declaration.
template <class T, class P, class Alloc> class AlexModelNode;
template <class T, class P> struct TraversalNode;

// A parent class for both types of ALEX nodes
template <class T, class P, class Alloc = std::allocator<std::pair<AlexKey<T>, P>>>
class AlexNode {
 public:

  typedef AlexNode<T, P, Alloc> self_type;
  typedef AlexModelNode<T, P, Alloc> model_node_type;

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
  LinearModel<T> model_ = LinearModel<T>((unsigned int) 1);

  // Node's linear model's key's max_length.
  unsigned int max_key_length_ = 1;

  // Could be either the expected or empirical cost, depending on how this field
  // is used
  double cost_ = 0.0;

  //parent of current node. Root is nullptr. Need to be given by parameter.
  model_node_type *parent_ = nullptr;

  // Variables for determining append-mostly behavior and checking errors.
  // max key in node, updates after inserts but not erases.
  AtomicVal<AlexKey<T> *>max_key_; 
  // min key in node, updates after inserts but not erases. 
  AtomicVal<AlexKey<T> *>min_key_;

  AlexNode() = default;
  explicit AlexNode(short level) : level_(level) {
    max_key_.val_ = new AlexKey<T>(1);
    min_key_.val_ = new AlexKey<T>(1);
    max_key_.val_->key_arr_[0] = STR_VAL_MIN;
    min_key_.val_->key_arr_[0] = STR_VAL_MAX;
  }
  AlexNode(short level, bool is_leaf) : is_leaf_(is_leaf), level_(level) {
    max_key_.val_ = new AlexKey<T>(1);
    min_key_.val_ = new AlexKey<T>(1);
    max_key_.val_->key_arr_[0] = STR_VAL_MIN;
    min_key_.val_->key_arr_[0] = STR_VAL_MAX;
  }
  AlexNode(short level, bool is_leaf, model_node_type *parent)
      : level_(level), is_leaf_(is_leaf), parent_(parent) {
    max_key_.val_ = new AlexKey<T>(1);
    min_key_.val_ = new AlexKey<T>(1);
    max_key_.val_->key_arr_[0] = STR_VAL_MIN;
    min_key_.val_->key_arr_[0] = STR_VAL_MAX;
  }
  AlexNode(short level, bool is_leaf, 
      model_node_type *parent, unsigned int max_key_length) 
        : is_leaf_(is_leaf), level_(level), model_(max_key_length),
          max_key_length_(max_key_length), parent_(parent) {
    max_key_.val_ = new AlexKey<T>(max_key_length_);
    min_key_.val_ = new AlexKey<T>(max_key_length_);
    if (typeid(T) != typeid(char)) { //numeric key
      std::fill(max_key_.val_->key_arr_, max_key_.val_->key_arr_ + max_key_length,
          std::numeric_limits<T>::lowest());
      std::fill(min_key_.val_->key_arr_, min_key_.val_->key_arr_ + max_key_length,
          std::numeric_limits<T>::max());
    }
    else { //string key
      max_key_.val_->key_arr_[0] = STR_VAL_MIN;
      std::fill(max_key_.val_->key_arr_ + 1, max_key_.val_->key_arr_ + max_key_length, 0);
      std::fill(min_key_.val_->key_arr_, min_key_.val_->key_arr_ + max_key_length,
          STR_VAL_MAX);
    }
  }

  AlexNode(self_type& other)
      : is_leaf_(other.is_leaf_),
        duplication_factor_(other.duplication_factor_),
        level_(other.level_),
        model_(other.model_),
        max_key_length_(other.max_key_length_),
        cost_(other.cost_),
        parent_(other.parent_) {
    max_key_.val_ = new AlexKey<T>(other.max_key_.val_->key_arr_, other.max_key_length_);
    min_key_.val_ = new AlexKey<T>(other.min_key_.val_->key_arr_, other.max_key_length_);
  }
  virtual ~AlexNode() = default;

  // The size in bytes of all member variables in this class
  virtual long long node_size() const = 0;
};

template <class T, class P, class Alloc = std::allocator<std::pair<AlexKey<T>, P>>>
class AlexModelNode : public AlexNode<T, P, Alloc> {
 public:
  AtomicVal<AlexNode<T, P, Alloc>**> children_ = AtomicVal<AlexNode<T, P, Alloc>**>(nullptr);
  int num_children_ = 0;

  typedef AlexNode<T, P, Alloc> basic_node_type;
  typedef AlexModelNode<T, P, Alloc> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<basic_node_type*>::other
      pointer_alloc_type;

  const Alloc& allocator_;

  std::map<uint32_t, basic_node_type**> old_childrens_;
  myLock old_childrens_lock;

  explicit AlexModelNode(const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(0, false), allocator_(alloc) {
      }

  explicit AlexModelNode(short level, const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(level, false), allocator_(alloc) {
      }

  explicit AlexModelNode(short level, self_type *parent,
                         const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(level, false, parent), allocator_(alloc){
      }
  
  explicit AlexModelNode(short level, self_type *parent,
                         unsigned int max_key_length, const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(level, false, parent, max_key_length), allocator_(alloc) {
      }

  ~AlexModelNode() {
    if (children_.val_ != nullptr) {
      pointer_allocator().deallocate(children_.val_, num_children_);
    }
    delete this->max_key_.val_;
    delete this->min_key_.val_;
  }

  AlexModelNode(const self_type& other)
      : AlexNode<T, P, Alloc>(other),
        allocator_(other.allocator_),
        num_children_(other.num_children_) {
    children_.val_ = new (pointer_allocator().allocate(other.num_children_))
        AlexNode<T, P, Alloc>*[other.num_children_];
    std::copy(other.children_.val_, 
              other.children_.val_ + other.num_children_,
              children_.val_);
  }

  self_type& operator=(const self_type& other) {
    this->is_leaf_ = other.is_leaf_;
    this->duplication_factor_ = other.duplication_factor_;
    this->level_ = other.level_;
    this->model_ = other.model_;
    this->max_key_length_ = other.max_key_length_;
    this->cost_ = other.cost_;
    this->parent_ = other.parent_;

    allocator_ = other.allocator_;

    pointer_allocator().deallocate(children_.val_, num_children_);
    children_.val_ = new AlexNode<T, P, Alloc>*[other.num_children_];
    std::copy(other.children_.val_, 
              other.children_.val_ + other.num_children_, 
              children_.val_);
    num_children_ = other.num_children_;
  }

  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  long long node_size() const override {
    long long size = sizeof(self_type);
    size += num_children_ * sizeof(AlexNode<T, P, Alloc>*);  // pointers to children
    return size;
  }

  // Helpful for debugging
  // This function is not synchronized. 
  // Call this only when your sure that model node is not being modified 
  bool validate_structure(bool verbose = false) const {
    //current model node metadata.
    LinearModel<T> *model_ = &(this->model_);
    AlexNode<T, P, Alloc>** children_ = children_.val_;

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
    for (int i = 0; i < model_->a_.max_key_length_; i++) {
      if (model_->a_[i] != 0.0) {zero_slope = 0; break;}
    }
    if (zero_slope) {
      if (verbose) {
        std::cout << "[Model node with zero slope] addr: " << this << ", level "
                  << this->level_ << std::endl;
      }
      return false;
    }

    AlexNode<T, P, Alloc>* cur_child = children_[0];
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
template <class T, class P, class Compare = AlexCompare,
          class Alloc = std::allocator<std::pair<AlexKey<T>, P>>,
          bool allow_duplicates = true>
class AlexDataNode : public AlexNode<T, P, Alloc> {
 public:
  typedef std::pair<AlexKey<T>, P> V;
  typedef std::pair<AlexKey<T>, AtomicVal<P>> AV;
  typedef AlexNode<T, P, Alloc> basic_node_type;
  typedef AlexModelNode<T, P, Alloc> model_node_type;
  typedef AlexDataNode<T, P, Compare, Alloc, allow_duplicates> self_type;
  typedef typename Alloc::template rebind<self_type>::other alloc_type;
  typedef typename Alloc::template rebind<P>::other payload_alloc_type;
  typedef typename Alloc::template rebind<AtomicVal<P>>::other atomic_payload_alloc_type;
  typedef typename Alloc::template rebind<V>::other value_alloc_type;
  typedef typename Alloc::template rebind<AV>::other atomic_value_alloc_type;
  typedef typename Alloc::template rebind<uint64_t>::other bitmap_alloc_type;
  typedef typename Alloc::template rebind<basic_node_type*>::other
      pointer_alloc_type;

  const Compare& key_less_;
  const Alloc& allocator_;

  // Forward declaration
  template <typename node_type = self_type, typename atomic_payload_return_type = AtomicVal<P>,
            typename atomic_value_return_type = AV>
  class Iterator;
  typedef Iterator<> iterator_type;
  typedef Iterator<const self_type, const P, const V> const_iterator_type;

  AtomicVal<self_type*> next_leaf_ = AtomicVal<self_type*>(nullptr);
  AtomicVal<self_type*> prev_leaf_ = AtomicVal<self_type*>(nullptr);
  AtomicVal<self_type*> pending_left_leaf_ = AtomicVal<self_type*>(nullptr);
  AtomicVal<self_type*> pending_right_leaf_ = AtomicVal<self_type*>(nullptr);

#if ALEX_DATA_NODE_SEP_ARRAYS
  AlexKey<T>* key_slots_ = nullptr;  // holds keys
  P* payload_slots_ =
      nullptr;  // holds payloads, must be same size as key_slots
#else
  AV* data_slots_ = nullptr;  // holds key-payload pairs
#endif

  AtomicVal<P> unused = AtomicVal<P>(0); // whether data node exists in alex or is about to be removed.
  struct RW_lock key_array_rw_lock = RW_lock();

  /* Below are unused attributes */
  //unsigned int max_key_length_ = 1; // maximum length of each key 
  //int key_type_ = DOUBLE; // key type for specific node.

  int data_capacity_ = 0;  // size of key/data_slots array
  int num_keys_ = 0;  // number of filled key/data slots (as opposed to gaps)
  T *the_max_key_arr_; //theoretic maximum key_arr
  T *the_min_key_arr_; //theoretic minimum key_arr

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
      sizeof(AV);  // cannot expand beyond this number of key/data slots

  // Counters used in cost models
  long long num_shifts_ = 0;                 // does not reset after resizing
  long long num_exp_search_iterations_ = 0;  // does not reset after resizing
  int num_lookups_ = 0;                      // does not reset after resizing
  int num_inserts_ = 0;                      // does not reset after resizing
  int num_resizes_ = 0;  // technically not required, but nice to have

  //now defined in all nodes.
  // Variables for determining append-mostly behavior
  // max key in node, updates after inserts but not erases.
  //AlexKey<T> *max_key_; 
  // min key in node, updates after inserts but not erases. 
  //AlexKey<T> *min_key_;

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
  AlexKey<T> kEndSentinel_; 

  /*** Constructors and destructors ***/

  AlexDataNode () : AlexNode<T, P, Alloc>(0, true) {
    T *kEndSentinel = new T[1];
    kEndSentinel_.key_arr_ = kEndSentinel;
    kEndSentinel_.max_key_length_ = 1;
    the_max_key_arr_ = new T[1];
    the_min_key_arr_ = new T[1];

    if (typeid(T) != typeid(char)) { //numeric key
      kEndSentinel[0] = STR_VAL_MAX;
      the_max_key_arr_[0] = std::numeric_limits<T>::max();
      the_min_key_arr_[0] = std::numeric_limits<T>::lowest();
    }
    else { // string key
      kEndSentinel[0] = STR_VAL_MAX;
      the_max_key_arr_[0] = STR_VAL_MAX;
      the_min_key_arr_[0] = STR_VAL_MIN;
    }
  }

  explicit AlexDataNode(unsigned int max_key_length, model_node_type *parent,
        const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(0, true, parent, max_key_length), key_less_(comp), allocator_(alloc) {
    T *kEndSentinel_arr = new T[max_key_length];
    kEndSentinel_.key_arr_ = kEndSentinel_arr;
    kEndSentinel_.max_key_length_ = max_key_length;
    the_max_key_arr_ = new T[max_key_length];
    the_min_key_arr_ = new T[max_key_length];

    if (typeid(T) != typeid(char)) { //numeric key
      std::fill(kEndSentinel_arr, kEndSentinel_arr + max_key_length,
          std::numeric_limits<T>::max());
      std::fill(the_max_key_arr_, the_max_key_arr_ + max_key_length,
          std::numeric_limits<T>::max());
      std::fill(the_min_key_arr_, the_min_key_arr_ + max_key_length,
          std::numeric_limits<T>::lowest());
    }
    else { //string key
      std::fill(kEndSentinel_arr, kEndSentinel_arr + max_key_length,
          STR_VAL_MAX);
      std::fill(the_max_key_arr_, the_max_key_arr_ + max_key_length,
          STR_VAL_MAX);
      the_min_key_arr_[0] = STR_VAL_MIN;
      std::fill(the_min_key_arr_ + 1, the_min_key_arr_ + max_key_length, 0);
    }
  }

  AlexDataNode(short level, int max_data_node_slots,
               unsigned int max_key_length, model_node_type *parent,
               const Compare& comp = Compare(), const Alloc& alloc = Alloc())
      : AlexNode<T, P, Alloc>(level, true, parent, max_key_length),
        key_less_(comp),
        allocator_(alloc),
        max_slots_(max_data_node_slots) {
    T *kEndSentinel_arr = new T[max_key_length];
    kEndSentinel_.key_arr_ = kEndSentinel_arr;
    kEndSentinel_.max_key_length_ = max_key_length;
    the_max_key_arr_ = new T[max_key_length];
    the_min_key_arr_ = new T[max_key_length];

    if (typeid(T) != typeid(char)) { //numeric key
      std::fill(kEndSentinel_arr, kEndSentinel_arr + max_key_length,
          std::numeric_limits<T>::max());
      std::fill(the_max_key_arr_, the_max_key_arr_ + max_key_length,
          std::numeric_limits<T>::max());
      std::fill(the_min_key_arr_, the_min_key_arr_ + max_key_length,
          std::numeric_limits<T>::lowest());
    }
    else { //string key
      std::fill(kEndSentinel_arr, kEndSentinel_arr + max_key_length,
          STR_VAL_MAX);
      std::fill(the_max_key_arr_, the_max_key_arr_ + max_key_length,
          STR_VAL_MAX);
      the_min_key_arr_[0] = STR_VAL_MIN;
      std::fill(the_min_key_arr_ + 1, the_min_key_arr_ + max_key_length, 0);
    }
  }

  ~AlexDataNode() {
#if ALEX_DATA_NODE_SEP_ARRAYS
    if (key_slots_ != nullptr) {
      delete[] key_slots_;
      payload_allocator().deallocate(payload_slots_, data_capacity_);
      bitmap_allocator().deallocate(bitmap_, bitmap_size_);
    }
#else
    if (data_slots_ != nullptr) {
      atomic_value_allocator().deallocate(data_slots_, data_capacity_);
      bitmap_allocator().deallocate(bitmap_, bitmap_size_);
    }
#endif
    delete this->min_key_.val_;
    delete this->max_key_.val_;
    delete[] the_max_key_arr_;
    delete[] the_min_key_arr_;
  }

  AlexDataNode(self_type& other)
      : AlexNode<T, P, Alloc>(other),
        key_less_(other.key_less_),
        allocator_(other.allocator_),
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
    the_max_key_arr_ = new T[other.max_key_length_];
    the_min_key_arr_ = new T[other.max_key_length_];
    std::copy(other.the_max_key_arr_, other.the_max_key_arr_ + other.max_key_length_,
        the_max_key_arr_);
    std::copy(other.the_min_key_arr_, other.the_min_key_arr_ + other.max_key_length_,
        the_min_key_arr_);
    kEndSentinel_ = other.kEndSentinel_;

    prev_leaf_.val_ = other.prev_leaf_.val_;
    next_leaf_.val_ = other.next_leaf_.val_;

#if ALEX_DATA_NODE_SEP_ARRAYS
    key_slots_ = new AlexKey<T>[other.data_capacity_]();
    std::copy(other.key_slots_, other.key_slots_ + other.data_capacity_,
              key_slots_);
    payload_slots_ = new (payload_allocator().allocate(other.data_capacity_))
        P[other.data_capacity_];
    std::copy(other.payload_slots_, other.payload_slots_ + other.data_capacity_,
              payload_slots_);
#else
    data_slots_ = new (atomic_value_allocator().allocate(other.data_capacity_))
        AV[other.data_capacity_];
    std::copy(other.data_slots_, other.data_slots_ + other.data_capacity_,
              data_slots_);
#endif
    bitmap_ = new (bitmap_allocator().allocate(other.bitmap_size_))
        uint64_t[other.bitmap_size_];
    std::copy(other.bitmap_, other.bitmap_ + other.bitmap_size_, bitmap_);
  }

  /*** Allocators ***/
  pointer_alloc_type pointer_allocator() {
    return pointer_alloc_type(allocator_);
  }

  payload_alloc_type payload_allocator() {
    return payload_alloc_type(allocator_);
  }

  atomic_payload_alloc_type atomic_payload_allocator() {
    return atomic_payload_alloc_type(allocator_);
  }

  value_alloc_type value_allocator() { return value_alloc_type(allocator_); }

  atomic_value_alloc_type atomic_value_allocator() {return atomic_value_alloc_type(allocator_);}

  bitmap_alloc_type bitmap_allocator() { return bitmap_alloc_type(allocator_); }

  /*** General helper functions ***/

  inline AlexKey<T>& get_key(int pos) const { return ALEX_DATA_NODE_KEY_AT(pos); }

  //newly added for actual content achieving without need for max_length data.
  inline T *get_key_arr(int pos) const { return get_key(pos).key_arr_; }

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
  T* first_key() const {
    for (int i = 0; i < data_capacity_; i++) {
      if (check_exists(i)) return get_key_arr(i);
    }
    return the_max_key_arr_;
  }

  // Value of last (i.e., max) key
  T* last_key() const {
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
  template <class K>
  forceinline bool key_less(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return key_less_(a, b);
  }

  // True if a <= b
  template <class K>
  forceinline bool key_lessequal(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return !key_less_(b, a);
  }

  // True if a > b
  template <class K>
  forceinline bool key_greater(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return key_less_(b, a);
  }

  // True if a >= b
  template <class K>
  forceinline bool key_greaterequal(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return !key_less_(a, b);
  }

  // True if a == b
  template <class K>
  forceinline bool key_equal(const AlexKey<T>& a, const AlexKey<K>& b) const {
    return !key_less_(a, b) && !key_less_(b, a);
  }

  /*** Iterator ***/

  // Forward iterator meant for iterating over a single data node.
  // By default, it is a "normal" non-const iterator.
  // Can be templated to be a const iterator.
  template <typename node_type, typename atomic_payload_return_type,
            typename atomic_value_return_type>
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
    AV operator*() const {
      return std::make_pair(node_->key_slots_[cur_idx_],
                            node_->payload_slots_[cur_idx_]);
    }
#else
    atomic_value_return_type& operator*() const {
      return node_->data_slots_[cur_idx_];
    }
#endif

    const AlexKey<T>& key() const {
#if ALEX_DATA_NODE_SEP_ARRAYS
      return node_->key_slots_[cur_idx_];
#else
      return node_->data_slots_[cur_idx_].first;
#endif
    }

    atomic_payload_return_type& payload() const {
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
      const V* values, int num_keys, double density, double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr, bool use_sampling = false,
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
    LinearModel<T> model;
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
                                  const LinearModel<T>* model) {
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
      const V* values, int num_keys, double density, double expected_insert_frac,
      const LinearModel<T>* existing_model = nullptr, DataNodeStats* stats = nullptr) {
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

    LinearModel<T> model;  // trained for full dense array
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
      LinearModel<T> sample_model(model.a_, model.b_, model.max_key_length_);
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
                                           const LinearModel<T>* sample_model) {
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
      const LinearModel<T>* existing_model = nullptr,
      DataNodeStats* stats = nullptr) {
    assert(left >= 0 && right <= node->data_capacity_);

    LinearModel<T> model;
    int num_actual_keys = 0;
    if (existing_model == nullptr) {
      model.max_key_length_ = node->max_key_length_;
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&model);
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      num_actual_keys = node->num_keys_in_range(left, right);
      model.max_key_length_ = existing_model->max_key_length_;
      model.a_ = new double[model.max_key_length_];
      for (unsigned int i = 0; i < model.max_key_length_; i++) {
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
                                                const LinearModel<T>* model) {
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
    key_slots_ = new AlexKey<T>[data_capacity_]();
    payload_slots_ =
        new (payload_allocator().allocate(data_capacity_)) P[data_capacity_];
#else
    data_slots_ =
        new (atomic_value_allocator().allocate(data_capacity_)) AV[data_capacity];
#endif
  }

  // Assumes pretrained_model is trained on dense array of keys
  // I also assumed that all DataNodes have properly initialized key length limit. (max_key_length_)
  // second condition must be handled when creating data node.
  void bulk_load(const V values[], int num_keys,
                 const LinearModel<T>* pretrained_model = nullptr,
                 bool train_with_sample = false) {
    /* minimal condition checking. */
    if (num_keys != 0) {assert(values[0].first.max_key_length_ == this->max_key_length_);}
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
      assert(pretrained_model->max_key_length_ == this->max_key_length_);
      for (unsigned int i = 0; i < pretrained_model->max_key_length_; i++) {
        this->model_.a_[i] = pretrained_model->a_[i];
      }
      this->model_.b_ = pretrained_model->b_;
    } else {
      build_model(values, num_keys, &(this->model_), train_with_sample);
    }
    this->model_.expand(static_cast<double>(data_capacity_) / num_keys);

#if DEBUG_PRINT
    for (int i = 0; i < num_keys; i++) {
      std::cout << values[i].first.key_arr_ << " is " << this->model_.predict_double(values[i].first) << std::endl;
    }
#endif

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

    std::copy(values[0].first.key_arr_, values[0].first.key_arr_ + this->max_key_length_,
      this->min_key_.val_->key_arr_);
    std::copy(values[num_keys-1].first.key_arr_, values[num_keys-1].first.key_arr_ + this->max_key_length_,
      this->max_key_.val_->key_arr_);
    
#if DEBUG_PRINT
      std::cout << values[0].first.key_arr_ << std::endl;
      std::cout << values[num_keys-1].first.key_arr_ << std::endl;
      std::cout << "with max length as " << this->max_key_length_ << std::endl;
      std::cout << "min_key_(data_node) : " << this->min_key_.val_->key_arr_ << std::endl;
      std::cout << "max_key_(data_node) : " << this->max_key_.val_->key_arr_ << std::endl;
#endif
  }

  // Bulk load using the keys between the left and right positions in
  // key/data_slots of an existing data node
  // keep_left and keep_right are set if the existing node was append-mostly
  // If the linear model and num_actual_keys have been precomputed, we can avoid
  // redundant work
  void bulk_load_from_existing(
      const self_type* node, int left, int right, uint32_t worker_id,
      bool keep_left = false, bool keep_right = false,
      const LinearModel<T>* precomputed_model = nullptr,
      int precomputed_num_actual_keys = -1) {
#if DEBUG_PRINT
    alex::coutLock.lock();
    if (left < 0) {
      std::cout << "t" << worker_id << " - ";
      std::cout <<"fucked left" << std::endl;
    }
    if (right > node->data_capacity_) {
      std::cout << "t" << worker_id << " - ";
      std::cout << "fucked right" << std::endl;}
    alex::coutLock.unlock();
#endif
    assert(left >= 0 && right <= node->data_capacity_);
    assert(node->max_key_length_ == this->max_key_length_);

    // Build model
    int num_actual_keys = 0;
    if (precomputed_model == nullptr || precomputed_num_actual_keys == -1) {
      const_iterator_type it(node, left);
      LinearModelBuilder<T> builder(&(this->model_));
      for (int i = 0; it.cur_idx_ < right && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
        num_actual_keys++;
      }
      builder.build();
    } else {
      assert(precomputed_model->max_key_length_ == this->max_key_length_);
      num_actual_keys = precomputed_num_actual_keys;
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
      this->min_key_.val_->key_arr_[i] = it.key().key_arr_[i];
    }
    for (; it.cur_idx_ < right && !it.is_end(); it++) {
      int position = this->model_.predict(it.key());
      position = std::max<int>(position, last_position + 1);

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
      this->max_key_.val_->key_arr_[i] = ALEX_DATA_NODE_KEY_AT(last_position).key_arr_[i];
    }

    expansion_threshold_ =
        std::min(std::max(data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(data_capacity_));
    contraction_threshold_ = data_capacity_ * kMinDensity_;
  }

  static void build_model(const V* values, int num_keys, LinearModel<T>* model,
                          bool use_sampling = false) {
    /* sampling only possible for integer, double type keys... for now.*/
    if (use_sampling && (model->max_key_length_ == 1)) {
      build_model_sampling(values, num_keys, model);
      return;
    }

    LinearModelBuilder<T> builder(model);
    for (int i = 0; i < num_keys; i++) {
      builder.add(values[i].first, i);
    }
    builder.build();
  }

  // Uses progressive non-random uniform sampling to build the model
  // Progressively increases sample size until model parameters are relatively
  // stable
  static void build_model_sampling(const V* values, int num_keys,
                                   LinearModel<T>* model,
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
    LinearModelBuilder<T> builder(model);
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
  inline int predict_position(const AlexKey<T>& key) const {
    int position = this->model_.predict(key);
    position = std::max<int>(std::min<int>(position, data_capacity_ - 1), 0);
    return position;
  }

  // Searches for the last non-gap position equal to key
  // If no positions equal to key, returns -1
  int find_key(const AlexKey<T>& key) {
    //start searching when no write is running.
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
  int find_lower(const AlexKey<T>& key) {
    num_lookups_++;
    int predicted_pos = predict_position(key);

    int pos = exponential_search_lower_bound(predicted_pos, key);
    return get_next_filled_position(pos, false);
  }

  // Searches for the first non-gap position greater than key
  // Returns position in range [0, data_capacity]
  // Compare with upper_bound()
  int find_upper(const AlexKey<T>& key) {
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
  std::pair<int, int> find_insert_position(const AlexKey<T>& key) {
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
  int upper_bound(const AlexKey<T>& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_upper_bound(position, key);
  }

  // Searches for the first position greater than key, starting from position m
  // Returns position in range [0, data_capacity]
  inline int exponential_search_upper_bound(int m, const AlexKey<T>& key) {
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
  inline int binary_search_upper_bound(int l, int r, const AlexKey<T>& key) const {
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
  int lower_bound(const AlexKey<T>& key) {
    num_lookups_++;
    int position = predict_position(key);
    return exponential_search_lower_bound(position, key);
  }

  // Searches for the first position no less than key, starting from position m
  // Returns position in range [0, data_capacity]
  inline int exponential_search_lower_bound(int m, const AlexKey<T>& key) {
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
  inline int binary_search_lower_bound(int l, int r, const AlexKey<T>& key) const {
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

  // First pair's first value in returned pair is fail flag:
  // 0 if successful insert , maybe with automatic expansion.
  // 1 if no insert because of significant cost deviation.
  // 2 if no insert because of "catastrophic" cost.
  // 3 if no insert because node is at max capacity.
  // -1 if key already exists and duplicates not allowed.
  //
  // First pair's second value in returned pair is position of inserted key, or of the
  // already-existing key.
  // -1 if no insertion.
  //
  // second pair has original data node pointer, and maybe new data node pointer
  std::pair<std::pair<int, int>, std::pair<self_type *, self_type *>> insert(
    const AlexKey<T>& key, const P& payload, 
    uint32_t worker_id, std::vector<TraversalNode<T,P>> *traversal_path = nullptr) {
    // Periodically check for catastrophe
#if DEBUG_PRINT
    alex::coutLock.lock();
    std::cout << "t" << worker_id << " - ";
    std::cout << "alex_nodes.h - expected_avg_shifts_ : " << expected_avg_shifts_ << std::endl;
    alex::coutLock.unlock();
#endif
    if (num_inserts_ % 64 == 0 && catastrophic_cost()) {
      return {{2, -1}, {this, nullptr}};
    }

    self_type *resized_data_node = nullptr;
    // Check if node is full (based on expansion_threshold)
    if (num_keys_ >= expansion_threshold_) {
      if (significant_cost_deviation()) {
        return {{1, -1}, {this, nullptr}};
      }
      if (catastrophic_cost()) {
        return {{2, -1}, {this, nullptr}};
      }
      if (num_keys_ > max_slots_ * kMinDensity_) {
        return {{3, -1}, {this, nullptr}};
      }
      // make new expanded node
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "alex_nodes.h insert : resizing data node" << std::endl;
      alex::coutLock.unlock();
#endif
      bool keep_left = is_append_mostly_right();
      bool keep_right = is_append_mostly_left();
      resized_data_node = resize(kMinDensity_, false, keep_left, keep_right);
      this->unused.val_ = 1;
      num_resizes_++;
    }

    int insertion_position = -1;

    if (resized_data_node) {
      //made new expanded node, should insert at that node.
      //also update model node's children_array.
      resized_data_node->unused.lock();
      int bucketID = traversal_path->back().bucketID;
      int repeats = 1 << this->duplication_factor_;
      int start_bucketID =
          bucketID - (bucketID % repeats);
      int end_bucketID = 
          start_bucketID + repeats;
      model_node_type *parent = resized_data_node->parent_;
      parent->children_.lock();
      basic_node_type** parent_new_children = new (pointer_allocator().allocate(parent->num_children_))
        basic_node_type*[parent->num_children_];
      basic_node_type** parent_old_children = parent->children_.val_;
      std::copy(parent_old_children, parent_old_children + parent->num_children_,
                parent_new_children);
      for (int i = start_bucketID; i < end_bucketID; i++) {
        parent_new_children[i] = resized_data_node;
      }
#if DEBUG_PRINT
          alex::coutLock.lock();
          std::cout << "t" << worker_id << " - ";
          std::cout << "alex_nodes.h resized node and updated children_" << std::endl;
          for (int i = 0; i < parent->num_children_; i++) {
            std::cout << i << " : " << parent_new_children[i] << std::endl;
          }
          alex::coutLock.unlock();
#endif
      parent->children_.val_ = parent_new_children;
      parent->children_.unlock();
      parent->old_childrens_lock.lock();
      parent->old_childrens_.insert({worker_id, parent_old_children});
      parent->old_childrens_lock.unlock();

      std::pair<int, int> positions = resized_data_node->find_insert_position(key);
      int upper_bound_pos = positions.second;
      if (!(allow_duplicates) && upper_bound_pos > 0 &&
          resized_data_node->key_equal(resized_data_node->key_slots_[upper_bound_pos - 1], key)) {
        return {{-1, upper_bound_pos - 1}, {this, resized_data_node}};
      }
      insertion_position = positions.first;
      if (insertion_position < resized_data_node->data_capacity_ &&
          !resized_data_node->check_exists(insertion_position)) {
        resized_data_node->insert_element_at(key, payload, insertion_position, 1);
      } else {
        insertion_position =
            resized_data_node->insert_using_shifts(key, payload, insertion_position);
      }
      // Update stats
      resized_data_node->num_keys_++;
      resized_data_node->num_inserts_++;
    }
    else {//should insert at current node.
#if DEBUG_PRINT
      alex::coutLock.lock();
      std::cout << "t" << worker_id << " - ";
      std::cout << "alex_nodes.h insert : resizing didn't happened and inserting." << std::endl;
      alex::coutLock.unlock();
#endif
      std::pair<int, int> positions = find_insert_position(key);
      int upper_bound_pos = positions.second;
      if (!allow_duplicates && upper_bound_pos > 0 &&
          key_equal(ALEX_DATA_NODE_KEY_AT(upper_bound_pos - 1), key)) {
        return {{-1, upper_bound_pos - 1}, {this, nullptr}};
      }
      insertion_position = positions.first;
      if (insertion_position < data_capacity_ &&
          !check_exists(insertion_position)) {
        insert_element_at(key, payload, insertion_position, 1);
      } else {
        insertion_position =
            insert_using_shifts(key, payload, insertion_position);
      }
      // Update stats
      num_keys_++;
      num_inserts_++;
    }
    
    return {{0, insertion_position}, {this, resized_data_node}};
  }

  // Resize the data node to the target density
  // For multithreading : makes new node with resized data node.
  self_type *resize(double target_density, bool force_retrain = false,
              bool keep_left = false, bool keep_right = false) {
    if (num_keys_ == 0) {
      return this;
    }

    self_type *new_data_node = new AlexDataNode(*this);
  
    //needs to connect new node with other data node
    self_type *this_prev_leaf_ = prev_leaf_.read();
    self_type *this_next_leaf_ = next_leaf_.read();
    if (this_prev_leaf_ != nullptr) {
      self_type *pl_pending_rl = this_prev_leaf_->pending_right_leaf_.read();
      if (pl_pending_rl != nullptr) {
        pl_pending_rl->next_leaf_.update(new_data_node);
        new_data_node->prev_leaf_.update(pl_pending_rl);
      }
      else {
        this_prev_leaf_->next_leaf_.update(new_data_node);
        new_data_node->prev_leaf_.update(this_prev_leaf_);
      }
    }
    if (this_next_leaf_ != nullptr) {
      self_type *nl_pending_ll = this_next_leaf_->pending_left_leaf_.read();
      if (nl_pending_ll != nullptr) {
        nl_pending_ll->prev_leaf_.update(new_data_node);
        new_data_node->next_leaf_.update(nl_pending_ll);
      }
      else {
        this_next_leaf_->prev_leaf_.update(new_data_node);
        new_data_node->next_leaf_.update(this_next_leaf_);
      }
    }

    int new_data_capacity =
        std::max(static_cast<int>(num_keys_ / target_density), num_keys_ + 1);
    auto new_bitmap_size =
        static_cast<size_t>(std::ceil(new_data_capacity / 64.));
    auto new_bitmap = new (bitmap_allocator().allocate(new_bitmap_size))
        uint64_t[new_bitmap_size]();  // initialize to all false
#if ALEX_DATA_NODE_SEP_ARRAYS
    AlexKey<T>* new_key_slots =
        new AlexKey<T>[new_data_capacity]();
    P* new_payload_slots = new (payload_allocator().allocate(new_data_capacity))
        P[new_data_capacity];
#else
    AV* new_data_slots = new (atomic_value_allocator().allocate(new_data_capacity))
        AV[new_data_capacity];
#endif

    // Retrain model if the number of keys is sufficiently small (under 50)
    if (num_keys_ < 50 || force_retrain) {
      const_iterator_type it(new_data_node, 0);
      LinearModelBuilder<T> builder(&(new_data_node->model_));
      for (int i = 0; it.cur_idx_ < data_capacity_ && !it.is_end(); it++, i++) {
        builder.add(it.key(), i);
      }
      builder.build();
      if (keep_left) {
        new_data_node->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
      } else if (keep_right) {
        new_data_node->model_.expand(static_cast<double>(data_capacity_) / num_keys_);
        new_data_node->model_.b_ += (new_data_capacity - data_capacity_);
      } else {
        new_data_node->model_.expand(static_cast<double>(new_data_capacity) / num_keys_);
      }
    } else {
      if (keep_right) {
        new_data_node->model_.b_ += (new_data_capacity - data_capacity_);
      } else if (!keep_left) {
        new_data_node->model_.expand(static_cast<double>(new_data_capacity) /
                            data_capacity_);
      }
    }

    int last_position = -1;
    int keys_remaining = num_keys_;
    const_iterator_type it(new_data_node, 0);
    for (; it.cur_idx_ < data_capacity_ && !it.is_end(); it++) {
      int position = new_data_node->model_.predict(it.key());
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
      new_key_slots[i] = new_data_node->kEndSentinel_;
#else
      new_data_slots[i].first = new_data_node->kEndSentinel;
#endif
    }

#if ALEX_DATA_NODE_SEP_ARRAYS
    delete[] new_data_node->key_slots_;
    new_data_node->payload_allocator().deallocate(
    new_data_node->payload_slots_, data_capacity_);
#else
    new_data_node->atomic_value_allocator().deallocate(
      new_data_node->data_slots_, data_capacity_);
#endif
    new_data_node->bitmap_allocator().deallocate(new_data_node->bitmap_, bitmap_size_);

    new_data_node->data_capacity_ = new_data_capacity;
    new_data_node->bitmap_size_ = new_bitmap_size;
#if ALEX_DATA_NODE_SEP_ARRAYS
    new_data_node->key_slots_ = new_key_slots;
    new_data_node->payload_slots_ = new_payload_slots;
#else
    data_slots_ = new_data_slots;
#endif
    new_data_node->bitmap_ = new_bitmap;

    new_data_node->expansion_threshold_ =
        std::min(std::max(new_data_node->data_capacity_ * kMaxDensity_,
                          static_cast<double>(num_keys_ + 1)),
                 static_cast<double>(new_data_node->data_capacity_));
    new_data_node->contraction_threshold_ = new_data_node->data_capacity_ * kMinDensity_;

    return new_data_node;
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
  // mode 0 : rw_lock already obtained, no need for another write wait (for insert_using_shifts)
  // mode 1 : rw_lock not obtained, need to do write wait (for other use cases)
  void insert_element_at(const AlexKey<T>& key, P payload, int pos, int mode = 0) {
    if (mode == 1) {key_array_rw_lock.write_wait();} //synchronization.
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
    if (mode == 1) {key_array_rw_lock.write_finished();}
  }

  // Insert key into pos, shifting as necessary in the range [left, right)
  // Returns the actual position of insertion
  int insert_using_shifts(const AlexKey<T>& key, P payload, int pos) {
    // Find the closest gap
    int gap_pos = closest_gap(pos);
    //std::cout << "gap pos is " << gap_pos << std::endl;
    set_bit(gap_pos);
    key_array_rw_lock.write_wait(); //for synchronization.
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
      key_array_rw_lock.write_finished();
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
      key_array_rw_lock.write_finished();
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

  /*** Stats ***/

  // Total size of node metadata
  long long node_size() const override { return sizeof(self_type); }

  // Total size in bytes of key/payload/data_slots and bitmap
  // NOTE THAT IT DOESN'T INCLUDE ALEX KEY'S POINTING ARRAY SIZE.
  long long data_size() const {
    long long data_size = data_capacity_ * sizeof(AlexKey<T>);
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
    AlexKey<T> end = ALEX_DATA_NODE_KEY_AT(data_capacity_ - 1);
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
  bool key_exists(const AlexKey<T>& key, bool validate_bitmap) const {
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
      AlexKey<T> cur_key = ALEX_DATA_NODE_KEY_AT(i);
      for (int j = 0; j < cur_key.max_key_length_; j++) {
        str += (std::to_string(cur_key.key_arr_[j]) + " ");
      }
      str += "\n";
    }
    return str;
  }
};

/* For use of alex
 * Save the traversal path down the RMI by having a linked list of these
 * structs. */
template<class T, class P>
struct TraversalNode {
  AlexModelNode<T,P>* node = nullptr;
  int bucketID = -1;
};

}