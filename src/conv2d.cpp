#include "tensor.h"
#include <iostream>
#include <cstring>

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

    Tensor output({N, OC, OH, OW});

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
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
                                        int ii = ((n * IC + ic_index) * IH + ih) * IW + iw;
                                        int wi = ((oc_index * group_ic + ic) * KH + kh) * KW + kw;
                                        sum += input.data()[ii] * weights.data()[wi];
                                    }
                                }
                            }
                        }
                        int oi = ((n * OC + oc_index) * OH + oh) * OW + ow;
                        output.data()[oi] = sum + bias.data()[oc_index];
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

    Tensor output({N, OC, OH, OW});

    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OC; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ++ic) {
                        int ih = oh * SH - PH;
                        int iw = ow * SW - PW;
                        if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                            int ii = ((n * IC + ic) * IH + ih) * IW + iw;
                            int wi = ((oc * IC + ic) * KH + 0) * KW + 0;
                            sum += input.data()[ii] * weights.data()[wi];
                        }
                    }
                    int oi = ((n * OC + oc) * OH + oh) * OW + ow;
                    output.data()[oi] = sum + bias.data()[oc];
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

    Tensor output({N, C, OH, OW});

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * SH - PH + kh * DH;
                            int iw = ow * SW - PW + kw * DW;
                            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                int ii = ((n * C + c) * IH + ih) * IW + iw;
                                int wi = ((c * 1 + 0) * KH + kh) * KW + kw;
                                sum += input.data()[ii] * weights.data()[wi];
                            }
                        }
                    }
                    int oi = ((n * C + c) * OH + oh) * OW + ow;
                    output.data()[oi] = sum + bias.data()[c];
                }
            }
        }
    }

    return output;
}
