// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "../core/alex_base.h"
#include "zipf.h"
#include <cctype>

bool load_binary_data(alex::AlexKey data[], int length, const std::string& file_path,
 unsigned int key_length, int key_type) {
  std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    return false;
  }

  if (key_type == 0) { //string key reading. NEED TO HANDLE MACRO PROBLEM.
    for (int i = 0; i < length; i++) {
      for (unsigned int pos = 0; pos < key_length; pos++) { 
        /* NOTE : I'M ASSUMING THAT BINARY STRING DATA FILES DO ZERO PADDING
         * SO IT COULD MAINTAIN THE CONSTRAINT OF MAXIMUM KEY_LENGTH 
         * WE MAY NEED TO FIX THIS LATER. */
        is.read(reinterpret_cast<char*>(&data[i].key_data_[pos]), std::streamsize(sizeof(char)));
      }
    }
  }
  else { //numeric key reading.
    for (int i = 0; i < length ; i++) {
      is.read(reinterpret_cast<char*>(data[i].key_data_), std::streamsize(sizeof(double)));
    }
  }
  is.close();
  return true;
}

bool load_text_data(alex::AlexKey array[], int length, const std::string& file_path,
 unsigned int key_length, int key_type) {
  std::ifstream is(file_path.c_str());
  if (!is.is_open()) {
    return false;
  }
  int i = 0;
  std::string str;
  
  if (key_type == 0) { //string key reading. NEED TO HANDLE MACRO PROBLEM. 
    while (std::getline(is, str) && i < length) {
      if (str.size() > key_length) { /* size above limit */
        return false;
      }
      for (unsigned int pos = 0; pos < str.size(); pos++) {
        array[i].key_data_[pos] = (double) tolower(str.at(pos));
      }
      i++;
    }
  }
  else { //numeric key reading.
    while (std::getline(is, str) && i < length) {
      std::istringstream ss(str);
      ss >> array[i].key_data_[0];
      i++;
    }
  }
  is.close();
  return true;
}

template <class T>
T* get_search_keys(T array[], int num_keys, int num_searches) {
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<int> dis(0, num_keys - 1);
  auto* keys = new T[num_searches];
  for (int i = 0; i < num_searches; i++) {
    int pos = dis(gen);
    keys[i] = array[pos];
  }
  return keys;
}

template <class T>
T* get_search_keys_zipf(T array[], int num_keys, int num_searches) {
  auto* keys = new T[num_searches];
  ScrambledZipfianGenerator zipf_gen(num_keys);
  for (int i = 0; i < num_searches; i++) {
    int pos = zipf_gen.nextValue();
    keys[i] = array[pos];
  }
  return keys;
}
