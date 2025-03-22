#include <benchmark/benchmark.h>
#include "operators.h"
#include "tensor.h"

static void BM_MatMul(benchmark::State& state) {
    int N = state.range(0);
    Tensor a({N, N});
    Tensor b({N, N});

    a.fillRandom();
    b.fillRandom();

    MatMul matmul;

    for (auto _ : state) {
        Tensor result = matmul.compute(a, b);
        benchmark::DoNotOptimize(result);
    }

    state.SetComplexityN(N);
}

BENCHMARK(BM_MatMul)->RangeMultiplier(2)->Range(64, 1024)->Complexity();
