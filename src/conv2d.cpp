#include "tensor.h"
#include <iostream>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
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
        for (int g = 0; g < groups; ++g) {
            #pragma omp parallel for
            for (int oc = 0; oc < group_oc; ++oc) {
                int oc_index = g * group_oc + oc;
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        float sum = 0.0f;
                        for (int ic = 0; ic < group_ic; ++ic) {
                            int ic_index = g * group_ic + ic;
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    int ih = oh * SH - PH + kh * DH;
                                    int iw = ow * SW - PW + kw * DW;
                                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                        int in_idx = n * IC * in_c_stride + ic_index * in_c_stride + ih * IW + iw;
                                        int w_idx = oc_index * group_ic * w_c_stride + ic * w_c_stride + kh * KW + kw;
                                        sum += input.data()[in_idx] * weights.data()[w_idx];
                                    }
                                }
                            }
                        }
                        int out_idx = n * OC * out_c_stride + oc_index * out_c_stride + oh * OW + ow;
                        output.data()[out_idx] = sum + bias.data()[oc_index];
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
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ic+=4) {
                        float acc = 0.0f;
                        for (int j = 0; j < 4 && (ic + j) < IC; ++j) {
                            int ih = oh * SH - PH;
                            int iw = ow * SW - PW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int in_idx = n * IC * in_c_stride + (ic + j) * in_c_stride + ih * IW + iw;
                                int w_idx = oc * IC + ic + j;
                                sum += input.data()[in_idx] * weights.data()[w_idx];
                            }
                        }
                        sum += acc;
                    }
                    int out_idx = n * OC * out_c_stride + oc * out_c_stride + oh * OW + ow;
                    output.data()[out_idx] = sum + bias.data()[oc];
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
                    float sum = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * SH - PH + kh * DH;
                            int iw = ow * SW - PW + kw * DW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int in_idx = n * C * in_c_stride + c * in_c_stride + ih * IW + iw;
                                int w_idx = c * w_c_stride + kh * KW + kw;
                                sum += input.data()[in_idx] * weights.data()[w_idx];
                            }
                        }
                    }
                    int out_idx = n * C * out_c_stride + c * out_c_stride + oh * OW + ow;
                    output.data()[out_idx] = sum + bias.data()[c];
                }
            }
        }
    }

    return output;
}
