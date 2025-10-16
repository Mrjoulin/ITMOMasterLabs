//
// Created by Matthew Ivanov on 25.09.2025.
//

#ifndef LAB1_HASHMAP_H
#define LAB1_HASHMAP_H

#include <vector>
#include <string>
#include <optional>
#include <iostream>

class HashMap {
private:
    enum EntryState {
        EMPTY,
        OCCUPIED,
        DELETED
    };

    struct Entry {
        int key;
        std::string value;
        EntryState state;

        Entry() : key(0), state(EMPTY) {}
    };

    std::vector<Entry> table;
    size_t element_count;
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.7;
    static constexpr double PHI = 0.618033988749895; // Golden ratio fraction: (sqrt(5) - 1) / 2

    // Probe sequence using linear probing
    [[nodiscard]] size_t probe(size_t index, size_t attempt) const;

    void resize(size_t size);

    // Internal methods
    void rehash();
    static size_t find_next_prime(size_t n);
    void insert_internal(int key, const std::string& value);

public:
    // Constructors
    HashMap();
    explicit HashMap(size_t initial_size);

    // Hash function
    [[nodiscard]] size_t hash(int key) const;

    void resize_prime(size_t size);

    // Core dictionary operations
    void insert(int key, const std::string& value);
    [[nodiscard]] std::optional<std::string> find(int key) const;
    bool remove(int key);
    [[nodiscard]] bool contains(int key) const;

    // Capacity and status
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] size_t capacity() const;
    [[nodiscard]] double load_factor() const;

    // Utility functions
    void display() const;

    // Rule of Five: Prevent copying and moving
    // HashMap(const HashMap&) = delete; // Copy Constructor
    // HashMap& operator=(const HashMap&) = delete; // Copy Assignment Operator
    // HashMap(HashMap&&) = delete; // Move Constructor
    // HashMap& operator=(HashMap&&) = delete; // Move Assignment Operator
};

#endif //LAB1_HASHMAP_H