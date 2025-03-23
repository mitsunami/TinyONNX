#include <benchmark/benchmark.h>
#include "tensor.h"
#include "operators.h"

static void BM_Conv2D(benchmark::State& state) {
    int N = 1;
    int IC = state.range(0);   // input channels
    int OC = state.range(1);   // output channels
    int H  = state.range(2);   // input height/width
    int K  = state.range(3);   // kernel size
    int stride = state.range(4);
    int pad    = state.range(5);
    int dilation = state.range(6);

    Tensor input({N, IC, H, H});
    Tensor weights({OC, IC, K, K});
    Tensor bias({OC});

    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;

    for (auto _ : state) {
        Tensor result = ops.conv2d(input, weights, bias, {stride, stride}, {pad, pad}, {dilation, dilation}, 1);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(int64_t(state.iterations()) * OC * H * H);
}

BENCHMARK(BM_Conv2D)
    ->Args({3, 32, 224, 3, 2, 1, 1})   // e.g. first layer in MobileNet
    ->Args({320, 1280, 7, 1, 1, 0, 1}) // MobileNet last Conv
    ->Args({32, 32, 28, 3, 1, 1, 1})   // Mid depthwise Conv
    ->Unit(benchmark::kMillisecond);
