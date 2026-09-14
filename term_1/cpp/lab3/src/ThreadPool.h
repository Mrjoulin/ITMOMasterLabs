#ifndef LAB3_THREADPOOL_H
#define LAB3_THREADPOOL_H

#include <vector>
#include <queue>
#include <thread>
#include <future>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <iostream>

#define THREAD_POOL_STATE_WORKING 0
#define THREAD_POOL_STATE_STOPPING 1
#define THREAD_POOL_STATE_STOPPED 2


class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    // Delete coping and moving
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template<typename T, typename Fn>
    std::future<T> dispatch_task(Fn && f);

    void stop();

private:
    void worker_thread();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::atomic<int> state_ {THREAD_POOL_STATE_WORKING}; // 0 working, 1 stopping, 2 stopped
    std::atomic<int> active_tasks_{0};
    std::atomic_flag queue_lock_ = ATOMIC_FLAG_INIT;
};

template<typename T, typename Fn>
std::future<T> ThreadPool::dispatch_task(Fn && f) {
    if (state_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("Cannot dispatch task on stopped ThreadPool");
    }
    // Create shared packaged_task for function
    auto task = std::make_shared<std::packaged_task<T()>>(std::forward<Fn>(f));
    // Get future from packaged_task
    std::future<T> result = task->get_future();
    // Capturing a spinlock to access the queue
    while (queue_lock_.test_and_set(std::memory_order_acquire)) {
        // Spin wait
        std::this_thread::yield();
    }
    // Add task to the queue
    tasks_.push([task]() { (*task)(); });
    // Release spinlock
    queue_lock_.clear(std::memory_order_release);

    return result;
}

#endif //LAB3_THREADPOOL_H