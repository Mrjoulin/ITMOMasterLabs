#include "ThreadPool.h"
#include <iostream>
#include <chrono>


ThreadPool::ThreadPool(const size_t num_threads) {
    const auto max_threads = std::thread::hardware_concurrency();
    const size_t n_thr = (num_threads > 0) ? std::min<size_t>(num_threads, max_threads) : max_threads;

    workers_.reserve(n_thr);
    for (size_t i = 0; i < n_thr; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::worker_thread() {
    while (state_.load(std::memory_order_relaxed) < THREAD_POOL_STATE_STOPPED) {
        // Capturing a spinlock to check the queue
        while (queue_lock_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (tasks_.empty()) {
            // Queue is empty, release spinlock
            queue_lock_.clear(std::memory_order_release);
            // Short sleep to reduce CPU usage
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            continue;
        }
        // Take next task from queue
        std::function<void()> task = std::move(tasks_.front());
        tasks_.pop();
        // Increment active tasks counter
        active_tasks_.fetch_add(1, std::memory_order_relaxed);
        // Release spinlock
        queue_lock_.clear(std::memory_order_release);
        // Complete the task
        try {
            task();
        } catch (...) {}
        // Reduce active tasks counter
        active_tasks_.fetch_sub(1, std::memory_order_relaxed);
    }
}

void ThreadPool::stop() {
    if (state_.load(std::memory_order_relaxed)) {
        return; // Already stopped
    }
    state_.store(THREAD_POOL_STATE_STOPPING, std::memory_order_relaxed);
    // Wait for all tasks to complete
    while (active_tasks_.load(std::memory_order_relaxed) > 0 ||
           queue_lock_.test(std::memory_order_relaxed) ||
           !tasks_.empty()) {
        std::this_thread::yield();
    }

    state_.store(THREAD_POOL_STATE_STOPPED, std::memory_order_relaxed);
    // Wait for all workers to stop
    for (auto& worker : workers_) worker.join();
    workers_.clear();
}