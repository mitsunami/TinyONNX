#include <benchmark/benchmark.h>
#include <xnnpack.h>
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
    int groups = 1;
    Tensor input({N, H, H, IC});
    Tensor weights({OC, K, K, IC/groups});
    Tensor bias({OC});

    input.fillRandom();
    weights.fillRandom();
    bias.fillRandom();

    Operators ops;

    xnn_status status = xnn_initialize(nullptr);
    pthreadpool_t pthreadpool_ = pthreadpool_create(0);

    for (auto _ : state) {
        Tensor result = ops.conv2d(
            input, weights, bias, 
            {K, K}, 
            {stride, stride}, 
            {pad, pad}, 
            {dilation, dilation}, 
            groups,
            pthreadpool_
        );
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(int64_t(state.iterations()) * OC * H * H);

    xnn_deinitialize();
    if (pthreadpool_) pthreadpool_destroy(pthreadpool_);
    
}

BENCHMARK(BM_Conv2D)
    ->Args({3, 32, 224, 3, 2, 1, 1})   // e.g. first layer in MobileNet
    ->Args({320, 1280, 7, 1, 1, 0, 1}) // MobileNet last Conv
    ->Args({32, 32, 28, 3, 1, 1, 1})   // Mid depthwise Conv
    ->Unit(benchmark::kMillisecond);
