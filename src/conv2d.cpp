#include "tensor.h"
#include <iostream>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif

Tensor conv2d_general(const Tensor& input, const Tensor& weights, const Tensor& bias,
                      const std::vector<int>& strides, const std::vector<int>& pads, const std::vector<int>& dilations, int groups) {
    const int N = input.shape()[0];
    const int IC = input.shape()[1];
    const int IH = input.shape()[2];
    const int IW = input.shape()[3];

    const int OC = weights.shape()[0];
    const int KH = weights.shape()[2];
    const int KW = weights.shape()[3];

    const int SH = strides[0];
    const int SW = strides[1];
    const int PH = pads[0];
    const int PW = pads[1];
    const int DH = dilations[0];
    const int DW = dilations[1];

    const int group_ic = IC / groups;
    const int group_oc = OC / groups;

    const int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    const int OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;

    const int in_c_stride = IH * IW;
    const int out_c_stride = OH * OW;
    const int w_c_stride = KH * KW;

    Tensor output({N, OC, OH, OW});

    for (int n = 0; n < N; ++n) {
        #pragma omp parallel for collapse(2)
        for (int g = 0; g < groups; ++g) {
            for (int oc = 0; oc < group_oc; ++oc) {
                int oc_index = g * group_oc + oc;
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
#ifdef __aarch64__
                        float32x4_t acc = vdupq_n_f32(0.0f);
#else
                        float sum = 0.0f;
#endif
                        for (int ic = 0; ic < group_ic; ++ic) {
                            int ic_index = g * group_ic + ic;
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    int ih = oh * SH - PH + kh * DH;
                                    int iw = ow * SW - PW + kw * DW;
                                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                        int in_idx = n * IC * in_c_stride + ic_index * in_c_stride + ih * IW + iw;
                                        int w_idx = oc_index * group_ic * w_c_stride + ic * w_c_stride + kh * KW + kw;
#ifdef __aarch64__
                                        float32x4_t in_val = vld1q_dup_f32(&in_data[in_idx]);
                                        float32x4_t w_val  = vld1q_dup_f32(&w_data[w_idx]);
                                        acc = vmlaq_f32(acc, in_val, w_val);
#else
                                        sum += input.data()[in_idx] * weights.data()[w_idx];
#endif
                                    }
                                }
                            }
                        }
#ifdef __aarch64__
                        float result = vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1) + vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3);
#else
                        float result = sum;
#endif
                        int out_idx = n * OC * out_c_stride + oc_index * out_c_stride + oh * OW + ow;
                        output.data()[out_idx] = result + bias.data()[oc_index];
                    }
                }
            }
        }
    }

    return output;
}

Tensor conv2d_pointwise(const Tensor& input, const Tensor& weights, const Tensor& bias,
                        const std::vector<int>& strides, int pad, int dilation) {
    const int N = input.shape()[0];
    const int IC = input.shape()[1];
    const int IH = input.shape()[2];
    const int IW = input.shape()[3];

    const int OC = weights.shape()[0];
    const int KH = weights.shape()[2];
    const int KW = weights.shape()[3];

    const int SH = strides[0];
    const int SW = strides[1];
    const int PH = pad;
    const int PW = pad;

    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;

    const int in_c_stride = IH * IW;
    const int out_c_stride = OH * OW;

    Tensor output({N, OC, OH, OW});

    for (int n = 0; n < N; ++n) {
        #pragma omp parallel for
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
#ifdef __aarch64__
                    float32x4_t acc = vdupq_n_f32(0.0f);
#else
                    float sum = 0.0f;
#endif
                    for (int ic = 0; ic < IC; ic+=4) {
                        for (int j = 0; j < 4 && (ic + j) < IC; ++j) {
                            int ih = oh * SH - PH;
                            int iw = ow * SW - PW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int in_idx = n * IC * in_c_stride + (ic + j) * in_c_stride + ih * IW + iw;
                                int w_idx = oc * IC + ic + j;
#ifdef __aarch64__
                                float32x4_t in_val = vld1q_dup_f32(&input.data()[in_idx]);
                                float32x4_t w_val  = vld1q_dup_f32(&weights.data()[w_idx]);
                                acc = vmlaq_f32(acc, in_val, w_val);
#else
                                sum += input.data()[in_idx] * weights.data()[w_idx];
#endif
                            }
                        }
                    }
#ifdef __aarch64__
                    float result = vaddvq_f32(acc);
#else
                    float result = sum;
#endif
                    int out_idx = n * OC * out_c_stride + oc * out_c_stride + oh * OW + ow;
                    output.data()[out_idx] = result + bias.data()[oc];
                }
            }
        }
    }

    return output;
}

Tensor conv2d_depthwise(const Tensor& input, const Tensor& weights, const Tensor& bias,
                        const std::vector<int>& strides,
                        const std::vector<int>& pads,
                        const std::vector<int>& dilations) {
    const int N = input.shape()[0];
    const int C = input.shape()[1]; // depthwise: IC == OC == groups
    const int IH = input.shape()[2];
    const int IW = input.shape()[3];

    const int KH = weights.shape()[2];
    const int KW = weights.shape()[3];

    const int SH = strides[0];
    const int SW = strides[1];
    const int PH = pads[0];
    const int PW = pads[1];
    const int DH = dilations[0];
    const int DW = dilations[1];

    const int OH = (IH + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    const int OW = (IW + 2 * PW - DW * (KW - 1) - 1) / SW + 1;

    const int in_c_stride = IH * IW;
    const int out_c_stride = OH * OW;
    const int w_c_stride = KH * KW;

    Tensor output({N, C, OH, OW});

    for (int n = 0; n < N; ++n) {
        #pragma omp parallel for
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
#ifdef __aarch64__
                    float32x4_t acc = vdupq_n_f32(0.0f);
#else
                    float sum = 0.0f;
#endif
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * SH - PH + kh * DH;
                            int iw = ow * SW - PW + kw * DW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int in_idx = n * C * in_c_stride + c * in_c_stride + ih * IW + iw;
                                int w_idx = c * w_c_stride + kh * KW + kw;
#ifdef __aarch64__
                                float32x4_t in_val = vld1q_dup_f32(&input.data()[in_idx]);
                                float32x4_t w_val = vld1q_dup_f32(&weights.data()[w_idx]);
                                acc = vmlaq_f32(acc, in_val, w_val);
#else
                                sum += input.data()[in_idx] * weights.data()[w_idx];
#endif
                            }
                        }
                    }
#ifdef __aarch64__
                    float result = vaddvq_f32(acc);
#else
                    float result = sum;
#endif
                    int out_idx = n * C * out_c_stride + c * out_c_stride + oh * OW + ow;
                    output.data()[out_idx] = result + bias.data()[c];
                }
            }
        }
    }

    return output;
}
