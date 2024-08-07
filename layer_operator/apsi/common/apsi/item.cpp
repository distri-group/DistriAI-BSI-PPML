// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// STD
#include <iterator>
#include <sstream>

// APSI
#include "apsi/item.h"
#include "apsi/util/utils.h"

// SEAL
#include "seal/util/blake2.h"

using namespace std;

namespace apsi {
    // Hashes input data to a fixed-size value using the Blake2b hashing algorithm
    void Item::hash_to_value(const void *in, size_t size)
    {
        // Hashes the input data 'in' of size 'size' to the 'value_' member variable using the Blake2b algorithm
        APSI_blake2b(value_.data(), sizeof(value_), in, size, nullptr, 0);
    }
    // Converts the Item to a Bitstring representation with a specified bit count
    Bitstring Item::to_bitstring(uint32_t item_bit_count) const
    {
        vector<unsigned char> bytes;
        bytes.reserve(sizeof(value_type));
        copy(value_.cbegin(), value_.cend(), back_inserter(bytes));
        return { move(bytes), item_bit_count };
    }

    string Item::to_string() const
    {
        return util::to_string(get_as<uint32_t>());
    }
} // namespace apsi
