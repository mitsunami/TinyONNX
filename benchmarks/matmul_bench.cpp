#include <benchmark/benchmark.h>
#include "operators.h"
#include "tensor.h"

static void BM_MatMul(benchmark::State& state) {
    int N = state.range(0);
    Tensor a({N, N});
    Tensor b({N, N});

    a.fillRandom();
    b.fillRandom();

    Operators ops;

    for (auto _ : state) {
        Tensor result = ops.matmul(a, b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(N);
}

BENCHMARK(BM_MatMul)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
