//
// Created by Matthew Ivanov on 25.09.2025.
//

#include "hashmap.h"

// Multiplication method for hashing
size_t HashMap::hash(const int key) const {
    const uint _key = static_cast<uint>(key + (1 << 31));
    const double irr_key = static_cast<double>(_key) * PHI;
    const double fractional = irr_key - static_cast<uint>(irr_key);
    return static_cast<size_t>(static_cast<double>(table.size()) * fractional);
}

// Probe sequence using linear probing
size_t HashMap::probe(const size_t index, const size_t attempt) const {
    return (index + attempt) % table.size();
}

// Reinitialize table with size = `size`
void HashMap::resize(const size_t size) {
    table.resize(size);
    element_count = 0;

    for (auto& entry : table) {
        entry = Entry();
    }
}

// Reinitialize table with size = next prime number after `size` (or to `size` if it's already prime)
void HashMap::resize_prime(const size_t size) {
    resize(find_next_prime(size));
}

void HashMap::rehash() {
    const std::vector<Entry> old_table = std::move(table);
    resize_prime(old_table.size() * 2 + 1);

    // Reinsert all non-deleted entries from old table
    for (const auto& entry : old_table) {
        if (entry.state == OCCUPIED) {
            insert_internal(entry.key, entry.value);
        }
    }
}

size_t HashMap::find_next_prime(size_t n) {
    if (n < 2) return 2;
    if (n < 4) return n;

    // All prime numbers has 1 or 5 in mod 6
    auto mod6 = static_cast<std::uint8_t>(n % 6);
    if (mod6 <= 1) { n += 1 - mod6; mod6 = 1; }
    else { n += 5 - mod6; mod6 = 5; }

    bool is_prime = false;
    const auto max_size = 2 * n; // Bertrand–Chebyshev theorem

    while (!is_prime && n < max_size) {
        is_prime = true;
        for(int i = 5; i * i <= n; i += 6)
        {
            if(n % i == 0 || n % (i + 2) == 0) {
                is_prime = false;
                if (mod6 == 1) { n += 4; mod6 = 5; }
                else { n += 2; mod6 = 1; }
                break;
            }
        }
    }
    return n;
}

void HashMap::insert_internal(const int key, const std::string& value) {
    const size_t index = hash(key);

    for (size_t attempt = 0; attempt < table.size(); ++attempt) {
        Entry& entry = table[probe(index, attempt)];

        if (entry.state == EMPTY || entry.state == DELETED) {
            entry.key = key;
            entry.value = value;
            entry.state = OCCUPIED;
            ++element_count;
            return;
        }

        if (entry.state == OCCUPIED  && entry.key == key) {
            entry.value = value; // Update existing key
            return;
        }
    }
    throw std::runtime_error("Hash table is full");
}


HashMap::HashMap() : element_count(0) {
    resize(2053);
}

HashMap::HashMap(const size_t initial_size) : element_count(0) {
    resize_prime(initial_size);
}

// Insert or update a key-value pair
void HashMap::insert(const int key, const std::string& value) {
    if (load_factor() >= LOAD_FACTOR_THRESHOLD) rehash();
    insert_internal(key, value);
}

// Find a value by key
std::optional<std::string> HashMap::find(const int key) const {
    if (table.empty()) return std::nullopt;

    const size_t index = hash(key);

    for (size_t attempt = 0; attempt < table.size(); ++attempt) {
        const Entry& entry = table[probe(index, attempt)];

        if (entry.state == EMPTY) {
            return std::nullopt; // Found empty slot, key doesn't exist
        }

        if (entry.state == OCCUPIED && entry.key == key) {
            return entry.value;
        }
    }

    return std::nullopt;
}

// Remove a key-value pair
bool HashMap::remove(const int key) {
    if (table.empty()) return false;

    const size_t index = hash(key);

    for (size_t attempt = 0; attempt < table.size(); ++attempt) {
        Entry& entry = table[probe(index, attempt)];

        if (entry.state == EMPTY) {
            return false; // Key not found
        }

        if (entry.state == OCCUPIED && entry.key == key) {
            entry.state = DELETED;
            entry.key = 0;
            entry.value.clear();
            // entry.value = std::string();
            --element_count;
            return true;
        }
    }

    return false;
}

// Check if key exists
bool HashMap::contains(const int key) const {
    return find(key).has_value();
}

// Get number of elements
size_t HashMap::size() const {
    return element_count;
}

// Check if HashMap is empty
bool HashMap::empty() const {
    return element_count == 0;
}

// Get current capacity
size_t HashMap::capacity() const {
    return table.size();
}

// Get load factor
double HashMap::load_factor() const {
    return table.empty() ? 0.0 : static_cast<double>(element_count) /  static_cast<double>(table.size());
}

// Display HashMap contents (for debugging)
void HashMap::display() const {
    std::cout << "HashMap (size: " << size() << ", capacity: " << capacity()
              << ", load factor: " << load_factor() << ")" << std::endl;

    for (size_t i = 0; i < table.size(); ++i) {
        const Entry& entry = table[i];
        std::cout << "[" << i << "]: ";

        if (entry.state == OCCUPIED) {
            std::cout << "Key: " << entry.key << ", Value: \"" << entry.value << "\"";
        } else if (entry.state == DELETED) {
            std::cout << "DELETED";
        } else {
            std::cout << "EMPTY";
        }
        std::cout << std::endl;
    }
}