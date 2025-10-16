#include "benchmark.h"

int main() {
    // Benchmark::benchmark_all_scenarios();
    // With step 50 from 100 to 1.000, step 500 from 1e3 to 1e4, step 5e3 from 1e4 to 1e5, step 5e4 from 1e5 to 1e6
    int step = 5;
    int iterations = 10;
    while (iterations < 1e6) {
        Benchmark::benchmark_all_scenarios(iterations);
        if (iterations / step >= 20) {
            step *= 10;
        }
        iterations += step;
    }
    return 0;
}